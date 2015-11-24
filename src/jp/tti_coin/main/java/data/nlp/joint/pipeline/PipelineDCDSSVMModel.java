package data.nlp.joint.pipeline;

import inference.State;

import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Vector;

import model.DCDSSVMModel;
import model.FeatureGenerator;
import model.SparseFeatureVector;
import config.Parameters;
import data.Data;
import data.Instance;
import data.Label;

// This is a naive implementation of the following TACL paper
// Chang, Ming-Wei, and Wen-tau Yih. 
// Dual Coordinate Descent Algorithms for Efficient Large Margin Structured Prediction.

public class PipelineDCDSSVMModel extends DCDSSVMModel {
	
	public PipelineDCDSSVMModel(Parameters params, FeatureGenerator fg) {
		super(params, fg);
	}
	
	private double sumAlpha(int index){
		if(!workingSet.containsKey(index))return 0.;
		double sum = 0.;		
		for(Example example:workingSet.get(index)){
			sum += example.getAlpha(); 
		}
		return sum;
	}	

	private double calcWeight(Data data, Label yGold, Label yStar){
		double c = 1.;
		double diff = 0.;
		for(int i = 0;i < yStar.size();i++){
			if(!yGold.getLabel(i).equals(yStar.getLabel(i))){
				c *= data.getLabelImportance(yGold.getLabel(i));
				c *= data.getLabelImportance(yStar.getLabel(i));
				diff++;
			}
		}
		assert c > 0.: c;
		return Math.pow(c, 1./diff*2.);
	}
	
	public void update(List<State> updates, boolean targetIsNE) {
		for(State yStarState:updates){
			Instance instance = yStarState.getInstance();
			int index = instance.getIndex();
			Label yGold = instance.getGoldLabel();
			Label yStar = yStarState.getLabel();
			if(yStarState.isCorrect()){
				assert sumAlpha(index) >= 0.;
				continue;
			}
			SparseFeatureVector fvGold  = ((PipelineFeatureGenerator)fg).calculateFeature(instance, yGold, yStar.size(), targetIsNE);
			SparseFeatureVector fvStar = ((PipelineFeatureGenerator)fg).calculateFeature(instance, yStar, yStar.size(), targetIsNE);

			SparseFeatureVector diff = new SparseFeatureVector(params);
			diff.add(1, fvGold);
			diff.add(-1., fvStar);
			diff.compact();
			double diffScore = evaluate(diff, false);
			double c = cParam;
			if(params.getUseWeighting()){
				c *= calcWeight(instance.getData(), yGold, yStar);
			}
			if((yStarState.getMargin() - diffScore - sumAlpha(index) / (2. * c)) > delta){
				if(!workingSet.containsKey(index)){
					workingSet.put(index, new Vector<Example>());
				}
				workingSet.get(index).add(0, new Example(yStarState, c));
			}
		}
		for(State state:updates){
			updateWeight(state.getInstance().getIndex());	
		}
	}

	public void updateWeight(int i, boolean targetIsNE) {
		if(!workingSet.containsKey(i))return;
		int size = workingSet.get(i).size();
		if(size < 1)return;
		List<Example> examples;
		if(size > 2){
			examples = workingSet.get(i).subList(1, size);
			Collections.shuffle(examples, rand);
			examples.add(0, workingSet.get(i).get(0));
			workingSet.put(i, examples);
		}else{
			examples = workingSet.get(i);
		}
		
		for(Iterator<Example> exampleIt = examples.iterator();exampleIt.hasNext();){
			Example example = exampleIt.next();
			Instance instance = example.getInstance();
			State yStarState = example.getState();
			double alpha = example.getAlpha();
			double c = example.getC();
			
			if(example.getDiffFv() == null){
				Label yStar = yStarState.getLabel();
				Label yGold = instance.getGoldLabel();			
				SparseFeatureVector fvStar = ((PipelineFeatureGenerator)fg).calculateFeature(instance, yStar, yStar.size(), targetIsNE);
				SparseFeatureVector fvGold = ((PipelineFeatureGenerator)fg).calculateFeature(instance, yGold, yStar.size(), targetIsNE);

				SparseFeatureVector diff = new SparseFeatureVector(params);
				diff.add(1, fvGold);
				diff.add(-1., fvStar);
				diff.compact();
				example.setDiffFv(diff);
			}
			SparseFeatureVector diff = example.getDiffFv();
			double d2 = yStarState.getMargin() - evaluate(diff, false) - sumAlpha(i) / (2. * c);
			if(d2 + (example.getAlpha() / (2. * c)) <= delta){
				exampleIt.remove();
				continue;
			}
			double norm = diff.getNorm();
			double d = d2 / (norm * norm + 1. / (2. * c));
			double newAlpha = Math.max(alpha+d, 0.);
			diff.addToWeight((newAlpha-alpha), weight);
			if(params.getUseAveraging()){
				diff.addToWeight((newAlpha-alpha) * trainStep, weightDiff);	
			}
			trainStep++;
			example.setAlpha(newAlpha);
			if(example.getAlpha() == 0.){
				exampleIt.remove();
			}
		}
		assert examples.size() == workingSet.get(i).size();
	}

}
