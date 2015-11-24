package model;

import inference.State;
import java.util.List;

import config.Parameters;
import data.Instance;
import data.Label;

public class PerceptronModel extends Model {
	public PerceptronModel(Parameters params, FeatureGenerator fg) {
		super(params, fg);
	}

	@Override
	public void update(List<State> updates) {
		SparseFeatureVector updateVector = new SparseFeatureVector(params);
		int numUpdates = 0;
		for(State yStarState:updates){
			Instance instance = yStarState.getInstance();	
			Label yGold = instance.getGoldLabel();	
			if(yStarState.isCorrect()){
				continue;
			}
			assert !yGold.equals(yStarState.getLabel()) : yGold.size();
			SparseFeatureVector fvGold = fg.calculateFeature(instance, yGold, yStarState.getLabel().size());
			double goldScore = evaluate(fvGold, false);
			if(yStarState.getScore()+yStarState.getMargin() <= goldScore){
				//beam search error!!
				continue;
			}
			SparseFeatureVector fvStar = fg.calculateFeature(instance, yStarState.getLabel(), yStarState.getLabel().size());
			updateVector.add(1., fvGold);
			updateVector.add(-1., fvStar);
			numUpdates++;
		}
		if(numUpdates > 0){
			updateVector.addToWeight(1. / params.getMiniBatch(), weight);
			if(params.getUseAveraging()){
				updateVector.addToWeight((double)trainStep / params.getMiniBatch(), weightDiff);
			}
			trainStep++;
		}
	}
}
