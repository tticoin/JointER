package model;

import inference.State;

import java.util.List;

import config.Parameters;
import data.Instance;
import data.Label;

public class SGDSVMModel extends Model {
	private final double eta0 = 1., mu0 = 1.;
	private final double lambda;
	private double wdiv;
	private double adiv;
	private double wfrac;

	public SGDSVMModel(Parameters params, FeatureGenerator fg) {
		super(params, fg);
		wdiv = adiv = 1.;
		wfrac = 0.;
		lambda = params.getLambda();
		assert lambda != 1.;
	}
	
	private void renorm(){
		if (wdiv != 1.0 || adiv != 1.0 || wfrac != 0.){
			if(params.getUseAveraging()){
				weightDiff.scale(1./adiv);
				weightDiff.add(wfrac/adiv, weight);
				adiv = 1.;
				wfrac = 0.;
			}
			weight.scale(1./wdiv);
			wdiv = 1.;
		}
	}
	
	public void averageWeight(){
		if(params.getUseAveraging()){
			renorm();
			this.aveWeight = new WeightVector(this.params);
			this.aveWeight.add(weightDiff);
		}
	}

	@Override
	public void update(List<State> updates) {
		SparseFeatureVector updateVector = new SparseFeatureVector(params);	
		double eta = eta0 / (1. + lambda * eta0 * (trainStep-1));
		wdiv /= (1. - eta * lambda);
		if (adiv > 1e5 || wdiv > 1e5) {
			renorm();
		}	
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
			SparseFeatureVector fvStar = fg.calculateFeature(instance, yStarState.getLabel(), yStarState.getLabel().size());
			// loss
			assert yStarState.getMargin() > 0.: yStarState.getMargin()+":"+yGold.getLabels()+":"+yStarState.getLabel().getLabels();
			double d = (goldScore - yStarState.getScore()) / wdiv > yStarState.getMargin() ? 0. : 1.;
			if(d == 0.){
				continue;
			}
			double alpha = eta * d * wdiv;
			assert alpha >= 0.;
			if (Math.abs(alpha) >= Double.MIN_VALUE){
				numUpdates++;
				updateVector.add(alpha, fvGold);
				updateVector.add(-alpha, fvStar);
			}
		}
		if(numUpdates > 0){
			updateVector.addToWeight(1. / params.getMiniBatch(), weight);
			double mu = mu0 / (1. + mu0 * (trainStep-1));
			if(params.getUseAveraging() && mu < 1.){
				updateVector.addToWeight(- wfrac / params.getMiniBatch(), weightDiff);
				adiv /= (1. - mu);
				wfrac +=  mu * adiv / wdiv;
			}
			trainStep++;
		}
	}
}
