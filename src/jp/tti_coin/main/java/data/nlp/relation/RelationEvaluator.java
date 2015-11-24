package data.nlp.relation;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;

import com.google.common.io.Files;

import model.Model;
import inference.Inference;
import inference.State;
import config.Parameters;
import data.Data;
import data.Instance;
import data.nlp.Document;
import data.nlp.Node;
import eval.Evaluator;

public class RelationEvaluator extends Evaluator {

	public RelationEvaluator(Parameters params, Inference infer) {
		super(params, infer);
	}
	
	private class Counter{
		int correct;
		int pred;
		int gold;
		public Counter(){
			correct = 0;
			pred = 0;
			gold = 0;
		}
		double getPrecision(){
			if(pred == 0)return 0.;
			return correct / (double)pred;
		}
		double getRecall(){
			if(gold == 0)return 0.;
			return correct / (double)gold;
		}
		double getF(){
			double precision = getPrecision();
			double recall = getRecall();
			if(precision + recall == 0.)return 0.;
			return 2 * precision * recall / (precision + recall);
		}
	}

	public void evaluate(Model model, Data valid){
		if(params.getVerbosity() == 0)return;
		int s_correct = 0;
		int t_sum = 0;
		int t_match = 0, correct = 0, pred = 0, gold = 0;
		Map<String, Counter> counters = new TreeMap<String, Counter>();
		for(int idx = 0;idx < valid.size();idx++){
			Instance instance = valid.getInstance(idx); 
			List<State> yStarStates = getInference().inferBestState(model, instance, true);
			instance = yStarStates.get(0).getInstance();
			if(yStarStates.get(0).getLabel().equals(instance.getGoldLabel())){
				s_correct++;
			}
			for(int i = 0;i < instance.size();i++){
				//TODO: label format
				String goldLabel = ((RelationLabelUnit)instance.getGoldLabel().getLabel(i)).getLabel().split("\\|")[0];
				String predLabel = ((RelationLabelUnit)yStarStates.get(0).getLabel().getLabel(i)).getLabel().split("\\|")[0];
				if(!counters.containsKey(goldLabel)){
					counters.put(goldLabel, new Counter());
				}
				if(!counters.containsKey(predLabel)){
					counters.put(predLabel, new Counter());
				}
				counters.get(goldLabel).gold++;
				counters.get(predLabel).pred++;
				if(instance.getGoldLabel().getLabel(i).equals(yStarStates.get(0).getLabel().getLabel(i))){
					if(!yStarStates.get(0).getLabel().getLabel(i).isNegative()){
						correct++;
						pred++;
						gold++;
					}
					counters.get(goldLabel).correct++;
					t_match++;
				}else{
					assert !yStarStates.get(0).getLabel().getLabel(i).isNegative() || !instance.getGoldLabel().getLabel(i).isNegative(); 
					if(!yStarStates.get(0).getLabel().getLabel(i).isNegative()){
						pred++;
					}
					if(!instance.getGoldLabel().getLabel(i).isNegative()){
						gold++;
					}
				}
				t_sum++;
			}
		}
		double precision = correct / (double)pred;
		double recall = correct / (double)gold;
		double f = 2. * precision * recall / (precision + recall);
		System.out.format("Accuracy (seq) : %.3f (%d/%d)", s_correct / (double)valid.size(), s_correct, valid.size());
		System.out.format(", Accuracy (tok) : %.3f (%d/%d)", t_match / (double)t_sum, t_match, t_sum);
		System.out.format(", (P, R, F) = (%.3f, %.3f, %.3f)", precision, recall, f);
		System.out.format(", (Correct, Prediction, Gold) = (%d, %d, %d)\n", correct, pred, gold);
		if(params.getVerbosity() > 3){
			for(Entry<String, Counter> counter:counters.entrySet()){
				System.out.format("%s: (P, R, F) = (%.3f, %.3f, %.3f)", counter.getKey(), counter.getValue().getPrecision(), counter.getValue().getRecall(), counter.getValue().getF());
				System.out.format(", (Correct, Prediction, Gold) = (%d, %d, %d)\n", counter.getValue().correct, counter.getValue().pred, counter.getValue().gold);
			}
		}
	}

	@Override
	public void predict(Model model, Data test) {
		assert test instanceof RelationData;
		try {
			for(Document document:((RelationData)test).getDocuments()){
				int relationId = 0;
				BufferedWriter annotationWriter = null;
				if(params.getUseByte()){
					annotationWriter = Files.newWriter(new File(document.getId()+params.getPredAnnotationExtension()), Charset.forName("US-ASCII"));
				}else{
					annotationWriter = Files.newWriter(new File(document.getId()+params.getPredAnnotationExtension()), Charset.forName("UTF-8"));
				}
				for(Instance instance:document.getInstances()){
					instance.cacheFeatures();
					List<State> yStarStates = getInference().inferBestState(model, instance, true);
					for(Node node:((RelationInstance)instance).getSentence().getGoldEntities().values()){
						annotationWriter.write(String.format("%s\t%s %d %d\t%s\n", node.getId(), node.getType(), node.getOffset().getStart(), node.getOffset().getEnd(), node.getText()));						
					}
					for(int i = 0;i < instance.size();i++){
						if(!yStarStates.get(0).getLabel().getLabel(i).isNegative()){
							EntityPair pair = (EntityPair)instance.getSequence().get(i);
							RelationLabelUnit labelUnit = (RelationLabelUnit)yStarStates.get(0).getLabel().getLabel(i);
							String[] labels = labelUnit.getLabel().split("\\|");
							assert labels.length >= 3: labelUnit.getLabel();
							Node e1, e2;
							if(labelUnit.isReverse()){
								e1 = pair.getE2();
								e2 = pair.getE1();			
							}else{
								e1 = pair.getE1();
								e2 = pair.getE2();
							}	
							for(int j = 0;j < labels.length - 2;j++){
								relationId++;
								String[] label = labels[j].split(":");
								assert label.length == 3: labels[j]+","+labelUnit.getLabel()+","+j+","+labels;
								annotationWriter.write(String.format("R%d\t%s %s:%s %s:%s\t\n", relationId, label[1], label[0], e1.getId(), label[2], e2.getId()));
							}
						}
					}
					instance.clearCachedFeatures();
				}
				annotationWriter.close();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
