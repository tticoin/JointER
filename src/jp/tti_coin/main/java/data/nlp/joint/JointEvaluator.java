package data.nlp.joint;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import utils.Tuple;

import com.google.common.collect.Multimap;
import com.google.common.collect.TreeMultimap;
import com.google.common.io.Files;

import model.Model;
import inference.Inference;
import inference.State;
import config.Parameters;
import data.Data;
import data.Instance;
import data.Label;
import data.LabelUnit;
import data.nlp.Document;
import data.nlp.Node;
import data.nlp.Offset;
import eval.Evaluator;

public class JointEvaluator extends Evaluator {

	public JointEvaluator(Parameters params, Inference infer) {
		super(params, infer);
	}

	private class Counter{
		public int correct;
		public int pred;
		public int gold;
		public Counter(){
			correct = 0;
			pred = 0;
			gold = 0;
		}
		public double getPrecision(){
			if(pred == 0)return 0.;
			return correct / (double)pred;
		}
		public double getRecall(){
			if(gold == 0)return 0.;
			return correct / (double)gold;
		}
		public double getF(){
			double precision = getPrecision();
			double recall = getRecall();
			if(precision + recall == 0.)return 0.;
			return 2 * precision * recall / (precision + recall);
		}
	}

	public void evaluate(Model model, Data valid){
		if(params.getVerbosity() == 0)return;
		evaluateAll(model, valid);
	}
	
	protected void getEntities(Multimap<String, Offset> entities, Map<String, Tuple<String, Offset>> idEntityMap, Instance instance, Label label){
		Map<Node, LabelUnit> nodeLabels = new TreeMap<Node, LabelUnit>();
		for(int i = 0;i < instance.size();i++){
			if(instance.getSequence().get(i) instanceof Word){
				nodeLabels.put(((Word)instance.getSequence().get(i)).getWord(), label.getLabel(i));
			}
		}
		int start = 0;
		int end = 0;
		for(Node node:nodeLabels.keySet()){
			if(nodeLabels.get(node).isNegative())continue;
			String position = ((WordLabelUnit)nodeLabels.get(node)).getPosition();
			if(position.equals(WordLabelUnit.B) || position.equals(WordLabelUnit.U)){
				start = node.getOffset().getStart();
			}
			if(position.equals(WordLabelUnit.L) || position.equals(WordLabelUnit.U)){
				end = node.getOffset().getEnd();
				WordLabelUnit unit = (WordLabelUnit)nodeLabels.get(node);
				String type = unit.getType();
				entities.put(type, new Offset(start, end));
				assert !idEntityMap.containsKey(node.getId());
				idEntityMap.put(node.getId(), new Tuple<String, Offset>(type, new Offset(start, end)));
			}
		}	
	}
	
	
	public void evaluateAll(Model model, Data valid){
		int ws_correct = 0;
		int wt_sum = 0;
		int wt_match = 0, wcorrect = 0, wpred = 0, wgold = 0;
		int we_correct = 0, we_pred = 0, we_gold = 0;
		Map<String, Counter> wcounters = new TreeMap<String, Counter>();
		Map<String, Counter> wentityCounters = new TreeMap<String, Counter>();
		
		int ps_correct = 0;
		int pt_sum = 0;
		int pt_match = 0, pcorrect = 0, ppred = 0, pgold = 0;
		//Map<String, Counter> wordCounters = new TreeMap<String, Counter>();
		Map<String, Counter> pcounters = new TreeMap<String, Counter>();
		
		int wps_correct = 0;
		int wpcorrect = 0, wppred = 0, wpgold = 0;
		//Map<String, Counter> wordCounters = new TreeMap<String, Counter>();
		Map<String, Counter> wpcounters = new TreeMap<String, Counter>();
		
		
		for(int idx = 0;idx < valid.size();idx++){
			Instance instance = valid.getInstance(idx); 
			List<State> yStarStates = getInference().inferBestState(model, instance, true);
			instance = yStarStates.get(0).getInstance();
			boolean wmatch = true;
			boolean pmatch = true;
			Multimap<String, Offset> goldEntities = TreeMultimap.create();
			Multimap<String, Offset> predEntities = TreeMultimap.create();
			Map<String, Tuple<String, Offset>> goldIdEntityMap = new TreeMap<String, Tuple<String, Offset>>();
			Map<String, Tuple<String, Offset>> predIdEntityMap = new TreeMap<String, Tuple<String, Offset>>();
			
			getEntities(goldEntities, goldIdEntityMap, instance, instance.getGoldLabel());
			getEntities(predEntities, predIdEntityMap, instance, yStarStates.get(0).getLabel());
			
			for(int i = 0;i < instance.size();i++){
				if(instance.getSequence().get(i) instanceof Word){
					String goldLabel = ((WordLabelUnit)instance.getGoldLabel().getLabel(i)).getLabel().split("\\|")[0];
					String predLabel = ((WordLabelUnit)yStarStates.get(0).getLabel().getLabel(i)).getLabel().split("\\|")[0];
					if(!wcounters.containsKey(goldLabel)){
						wcounters.put(goldLabel, new Counter());
					}
					if(!wcounters.containsKey(predLabel)){
						wcounters.put(predLabel, new Counter());
					}
					wcounters.get(goldLabel).gold++;
					wcounters.get(predLabel).pred++;
					if(instance.getGoldLabel().getLabel(i).equals(yStarStates.get(0).getLabel().getLabel(i))){
						if(!yStarStates.get(0).getLabel().getLabel(i).isNegative()){
							wcorrect++;
							wpred++;
							wgold++;
						}
						wcounters.get(goldLabel).correct++;
						wt_match++;
					}else{
						wmatch = false;
						assert !yStarStates.get(0).getLabel().getLabel(i).isNegative() || !instance.getGoldLabel().getLabel(i).isNegative():
							((WordLabelUnit)yStarStates.get(0).getLabel().getLabel(i)).getLabel()+":"+((WordLabelUnit)instance.getGoldLabel().getLabel(i)).getLabel();
						if(!yStarStates.get(0).getLabel().getLabel(i).isNegative()){
							wpred++;
						}
						if(!instance.getGoldLabel().getLabel(i).isNegative()){
							wgold++;
						}
					}
					wt_sum++;
				}else if(instance.getSequence().get(i) instanceof Pair){						
					PairLabelUnit goldLabelUnit = (PairLabelUnit)instance.getGoldLabel().getLabel(i);
					String goldType = goldLabelUnit.getType();
					PairLabelUnit predLabelUnit = (PairLabelUnit)yStarStates.get(0).getLabel().getLabel(i);
					String predType = predLabelUnit.getType();
					if(!pcounters.containsKey(goldType)){
						pcounters.put(goldType, new Counter());
					}
					if(!pcounters.containsKey(predType)){
						pcounters.put(predType, new Counter());
					}
					pcounters.get(goldType).gold++;
					pcounters.get(predType).pred++;
					if(instance.getGoldLabel().getLabel(i).equals(yStarStates.get(0).getLabel().getLabel(i))){
						if(!yStarStates.get(0).getLabel().getLabel(i).isNegative()){
							pcorrect++;
							ppred++;
							pgold++;
						}
						pcounters.get(goldType).correct++;
						pt_match++;
					}else{
						pmatch = false;
						assert !yStarStates.get(0).getLabel().getLabel(i).isNegative() || !instance.getGoldLabel().getLabel(i).isNegative():
							((PairLabelUnit)yStarStates.get(0).getLabel().getLabel(i)).getLabel()+":"+((PairLabelUnit)instance.getGoldLabel().getLabel(i)).getLabel();
						if(!yStarStates.get(0).getLabel().getLabel(i).isNegative()){
							ppred++;
						}
						if(!instance.getGoldLabel().getLabel(i).isNegative()){
							pgold++;
						}
					}
					pt_sum++;
					
					Pair pair = (Pair)instance.getSequence().get(i);
					String w1Id = pair.getW1().getWord().getId();
					String w2Id = pair.getW2().getWord().getId();
					Tuple<String, Offset> goldE1 = goldIdEntityMap.get(w1Id);
					Tuple<String, Offset> goldE2 = goldIdEntityMap.get(w2Id);
					Tuple<String, Offset> predE1 = predIdEntityMap.get(w1Id);
					Tuple<String, Offset> predE2 = predIdEntityMap.get(w2Id);				
					if(!wpcounters.containsKey(goldType)){
						wpcounters.put(goldType, new Counter());
					}
					if(!wpcounters.containsKey(predType)){
						wpcounters.put(predType, new Counter());
					}
					wpcounters.get(goldType).gold++;
					wpcounters.get(predType).pred++;
					assert instance.getGoldLabel().getLabel(i).isNegative() || (goldE1 != null && goldE2 != null);
					assert yStarStates.get(0).getLabel().getLabel(i).isNegative() || (predE1 != null && predE2 != null);
					if(instance.getGoldLabel().getLabel(i).equals(yStarStates.get(0).getLabel().getLabel(i)) &&
							((goldE1 == predE1) || (goldE1 != null && predE1 != null && goldE1.getT1().equals(predE1.getT1()) && goldE1.getT2().equals(predE1.getT2()))) && 
							((goldE2 == predE2) || (goldE2 != null && predE2 != null && goldE2.getT1().equals(predE2.getT1()) && goldE2.getT2().equals(predE2.getT2())))){
						if(!yStarStates.get(0).getLabel().getLabel(i).isNegative()){
							wpcorrect++;
							wppred++;
							wpgold++;
						}
						wpcounters.get(goldType).correct++;
					}else{
						if(!yStarStates.get(0).getLabel().getLabel(i).isNegative()){
							wppred++;
						}
						if(!instance.getGoldLabel().getLabel(i).isNegative()){
							wpgold++;
						}
					}
				}
			}
			if(wmatch){
				ws_correct++;
			}
			if(pmatch){
				ps_correct++;
			}
			if(wmatch && pmatch){
				wps_correct++;
			}
			for(String type:goldEntities.keySet()){
				if(!wentityCounters.containsKey(type)){
					wentityCounters.put(type, new Counter());
				}
				int ngolds = goldEntities.get(type).size();
				wentityCounters.get(type).gold += ngolds; 
				we_gold += ngolds;
			}				
			for(String type:predEntities.keySet()){
				if(!wentityCounters.containsKey(type)){
					wentityCounters.put(type, new Counter());
				}
				int npreds = predEntities.get(type).size();
				wentityCounters.get(type).pred += npreds;
				we_pred += npreds;
			}
			Set<String> overlappingTypes = new TreeSet<String>(goldEntities.keySet());
			overlappingTypes.retainAll(predEntities.keySet());
			for(String type:overlappingTypes){
				Set<Offset> overlappingEntities = new TreeSet<Offset>(goldEntities.get(type));
				overlappingEntities.retainAll(predEntities.get(type));
				int ncorrect = overlappingEntities.size();
				wentityCounters.get(type).correct += ncorrect;
				we_correct += ncorrect;
			}
		}
		{
			double wprecision = wcorrect / (double)wpred;
			double wrecall = wcorrect / (double)wgold;
			double wf = 2. * wprecision * wrecall / (wprecision + wrecall);
			System.out.format("=============== WORD ===============\n");
			System.out.format("Accuracy (seq) : %.3f (%d/%d)", ws_correct / (double)valid.size(), ws_correct, valid.size());
			System.out.format(", Accuracy (tok) : %.3f (%d/%d)\n", wt_match / (double)wt_sum, wt_match, wt_sum);
			System.out.format("Tok : (P, R, F) = (%.3f, %.3f, %.3f)", wprecision, wrecall, wf);
			System.out.format(", (Correct, Prediction, Gold) = (%d, %d, %d)\n", wcorrect, wpred, wgold);
			if(params.getVerbosity() > 2){
				for(Entry<String, Counter> counter:wcounters.entrySet()){
					System.out.format("%s : (P, R, F) = (%.3f, %.3f, %.3f)", counter.getKey(), counter.getValue().getPrecision(), counter.getValue().getRecall(), counter.getValue().getF());
					System.out.format(", (Correct, Prediction, Gold) = (%d, %d, %d)\n", counter.getValue().correct, counter.getValue().pred, counter.getValue().gold);
				}
			}
		}
		{
			double we_precision = we_correct / (double)we_pred;
			double we_recall = we_correct / (double)we_gold;
			double we_f = 2. * we_precision * we_recall / (we_precision + we_recall);
			System.out.format("Entity : (P, R, F) = (%.3f, %.3f, %.3f)", we_precision, we_recall, we_f);
			System.out.format(", (Correct, Prediction, Gold) = (%d, %d, %d)\n", we_correct, we_pred, we_gold);
			if(params.getVerbosity() > 2){
				for(Entry<String, Counter> counter:wentityCounters.entrySet()){
					System.out.format("%s: (P, R, F) = (%.3f, %.3f, %.3f)", counter.getKey(), counter.getValue().getPrecision(), counter.getValue().getRecall(), counter.getValue().getF());
					System.out.format(", (Correct, Prediction, Gold) = (%d, %d, %d)\n", counter.getValue().correct, counter.getValue().pred, counter.getValue().gold);
				}
			}
		}
		{
			double pprecision = pcorrect / (double)ppred;
			double precall = pcorrect / (double)pgold;
			double pf = 2. * pprecision * precall / (pprecision + precall);
			System.out.format("=============== PAIR ===============\n");
			System.out.format("Accuracy (seq) : %.3f (%d/%d)", ps_correct / (double)valid.size(), ps_correct, valid.size());
			System.out.format(", Accuracy (tok) : %.3f (%d/%d)\n", pt_match / (double)pt_sum, pt_match, pt_sum);
			System.out.format("ALL : (P, R, F) = (%.3f, %.3f, %.3f)", pprecision, precall, pf);
			System.out.format(", (Correct, Prediction, Gold) = (%d, %d, %d)\n", pcorrect, ppred, pgold);
			if(params.getVerbosity() > 2){
				for(Entry<String, Counter> counter:pcounters.entrySet()){
					System.out.format("%s : (P, R, F) = (%.3f, %.3f, %.3f)", counter.getKey(), counter.getValue().getPrecision(), counter.getValue().getRecall(), counter.getValue().getF());
					System.out.format(", (Correct, Prediction, Gold) = (%d, %d, %d)\n", counter.getValue().correct, counter.getValue().pred, counter.getValue().gold);
				}
			}
		}
		{
			double wpprecision = wpcorrect / (double)wppred;
			double wprecall = wpcorrect / (double)wpgold;
			double wpf = 2. * wpprecision * wprecall / (wpprecision + wprecall);
			System.out.format("=============== WORDPAIR ===============\n");
			System.out.format("Accuracy (seq) : %.3f (%d/%d)\n", wps_correct / (double)valid.size(), wps_correct, valid.size());
			System.out.format("ALL : (P, R, F) = (%.3f, %.3f, %.3f)", wpprecision, wprecall, wpf);
			System.out.format(", (Correct, Prediction, Gold) = (%d, %d, %d)\n", wpcorrect, wppred, wpgold);
			if(params.getVerbosity() > 2){
				for(Entry<String, Counter> counter:wpcounters.entrySet()){
					System.out.format("%s : (P, R, F) = (%.3f, %.3f, %.3f)", counter.getKey(), counter.getValue().getPrecision(), counter.getValue().getRecall(), counter.getValue().getF());
					System.out.format(", (Correct, Prediction, Gold) = (%d, %d, %d)\n", counter.getValue().correct, counter.getValue().pred, counter.getValue().gold);
				}
			}
		}
	}
	
	private Map<Node, String> getGoldNodeStrings(Instance instance){
		Map<Node, String> goldLabels = new TreeMap<Node, String>();
		Map<Node, LabelUnit> nodeLabels = new TreeMap<Node, LabelUnit>();
		for(int i = 0;i < instance.size();i++){
			if(instance.getSequence().get(i) instanceof Word){
				nodeLabels.put(((Word)instance.getSequence().get(i)).getWord(), instance.getGoldLabel().getLabel(i));
			}
		}
		int start = 0;
		int end = 0;
		for(Node node:nodeLabels.keySet()){
			if(nodeLabels.get(node).isNegative())continue;
			String position = ((WordLabelUnit)nodeLabels.get(node)).getPosition();
			if(position.equals(WordLabelUnit.B) || position.equals(WordLabelUnit.U)){
				start = node.getOffset().getStart();
			}
			if(position.equals(WordLabelUnit.L) || position.equals(WordLabelUnit.U)){
				end = node.getOffset().getEnd();
				String type = ((WordLabelUnit)nodeLabels.get(node)).getType();
				goldLabels.put(node, String.format("%s-%d-%d", type, start, end));
			}
		}
		return goldLabels;
	}

	@Override
	public void predict(Model model, Data test) {
		assert test instanceof JointData;
		try {
			for(Document document:((JointData)test).getDocuments()){
				int nodeId = 0;
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
					//annotationWriter.write(String.format("%s\t%s %d %d\t%s\n", node.getId(), node.getType(), node.getOffset().getStart(), node.getOffset().getEnd(), node.getText()));						
					

					Map<Node, LabelUnit> nodeLabels = new TreeMap<Node, LabelUnit>();
					for(int i = 0;i < instance.size();i++){
						if(instance.getSequence().get(i) instanceof Word){
							nodeLabels.put(((Word)instance.getSequence().get(i)).getWord(), yStarStates.get(0).getLabel().getLabel(i));
						}
					}
					int start = 0;
					int end = 0;
					Map<Node, Integer> nodeIds = new TreeMap<Node, Integer>();
					Map<Integer, String> nodeStrings = new TreeMap<Integer, String>();
					for(Node node:nodeLabels.keySet()){
						if(nodeLabels.get(node).isNegative())continue;
						String position = ((WordLabelUnit)nodeLabels.get(node)).getPosition();
						if(position.equals(WordLabelUnit.B) || position.equals(WordLabelUnit.U)){
							start = node.getOffset().getStart();
						}
						if(position.equals(WordLabelUnit.L) || position.equals(WordLabelUnit.U)){
							end = node.getOffset().getEnd();
							nodeId++;
							String type = ((WordLabelUnit)nodeLabels.get(node)).getType();
							annotationWriter.write(String.format("T%d\t%s %d %d\t%s\n", nodeId, type, start, end, node.getDocument().getText(new Offset(start, end))));						
							nodeIds.put(node, nodeId);
							nodeStrings.put(nodeId, String.format("%s-%d-%d", type, start, end));
						}
					}
					
					Map<Node, String> goldNodeStrings = getGoldNodeStrings(instance);
					Set<String> goldRelStrings = new TreeSet<String>();
					for(int i = 0;i < instance.size();i++){
						if(!(instance.getSequence().get(i) instanceof Pair)){
							continue;
						}
						if(!instance.getGoldLabel().getLabel(i).isNegative()){
							Pair pair = (Pair)instance.getSequence().get(i);
							PairLabelUnit labelUnit = (PairLabelUnit)instance.getGoldLabel().getLabel(i);
							String[] labels = labelUnit.getLabel().split("\\|");
							Node w1 = pair.getW1().getWord();
							Node w2 = pair.getW2().getWord();
							for(int j = 0;j < labels.length;j++){
								if(labels[j].isEmpty())continue;
								String[] label = labels[j].split(":");
								assert label.length == 3: labels[j]+","+labelUnit.getLabel()+","+j+","+labels;
								String w1String = goldNodeStrings.get(w1);
								String w2String = goldNodeStrings.get(w2);
								if(params.getVerbosity() > 3){
									if(label[0].compareTo(label[2]) < 0){
										goldRelStrings.add(String.format("REL %s %s %s:%s %s:%s", document.getId(), label[1], label[0], w1String, label[2], w2String));
									}else{
										goldRelStrings.add(String.format("REL %s %s %s:%s %s:%s", document.getId(), label[1], label[2], w2String, label[0], w1String));
									}								
								}
							}
						}
					}					
					
					for(int i = 0;i < instance.size();i++){
						if(!(instance.getSequence().get(i) instanceof Pair)){
							continue;
						}
						if(!yStarStates.get(0).getLabel().getLabel(i).isNegative()){
							Pair pair = (Pair)instance.getSequence().get(i);
							PairLabelUnit labelUnit = (PairLabelUnit)yStarStates.get(0).getLabel().getLabel(i);
							String[] labels = labelUnit.getLabel().split("\\|");
							Node w1 = pair.getW1().getWord();
							Node w2 = pair.getW2().getWord();
							for(int j = 0;j < labels.length;j++){
								if(labels[j].isEmpty())continue;
								relationId++;
								String[] label = labels[j].split(":");
								assert label.length == 3: labels[j]+","+labelUnit.getLabel()+","+j+","+labels;
								int w1Id = nodeIds.get(w1);
								int w2Id = nodeIds.get(w2);
								annotationWriter.write(String.format("R%d\t%s %s:T%d %s:T%d\t\n", relationId, label[1], label[0], w1Id, label[2], w2Id));
								if(params.getVerbosity() > 3){
									String relString = "";
									if(label[0].compareTo(label[2]) < 0){
										relString = String.format("REL %s %s %s:%s %s:%s", document.getId(), label[1], label[0], nodeStrings.get(w1Id), label[2], nodeStrings.get(w2Id));
									}else{
										relString = String.format("REL %s %s %s:%s %s:%s", document.getId(), label[1], label[2], nodeStrings.get(w2Id), label[0], nodeStrings.get(w1Id));
									}						
									if(goldRelStrings.contains(relString)){
										System.out.println("TP "+relString);
										goldRelStrings.remove(relString);
									}else{
										System.out.println("FP "+relString);
									}
								}
							}
						}
						instance.clearCachedFeatures();
					}
					for(String relString:goldRelStrings){
						System.out.println("FN "+relString);
					}
				}
				annotationWriter.close();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
