package model;

import inference.State;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.List;

import com.google.common.io.Files;
import config.Parameters;

import data.Instance;
import data.Label;

// This is an implementaion of AdaGrad.
// Duchi, John, Elad Hazan, and Yoram Singer. 
// Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.
// Journal of Machine Learning Research 12 (2011): 2121-2159.

public class AdaGradModel extends Model {
	protected WeightVector featureSquaredCounts;
	protected final double eta = 1.;
	protected final double delta = .1;
	protected final double lambda;

	public AdaGradModel(Parameters params, FeatureGenerator fg) {
		super(params, fg);
		featureSquaredCounts = new WeightVector(params);
		lambda = params.getLambda();
	}

	private void updateSquaredCounts(SparseFeatureVector fv) {
		double[] counts = featureSquaredCounts.get();
		for(VectorInfo vector:fv.getFeatureVectors()){
			int[] index = vector.getSv().getIndex();
			double[] grad = vector.getSv().getData();
			double scale = vector.getWeight();
			int size = vector.getSv().getUsed();
			scale = scale * scale;
			for(int i = 0;i < size;i++){
				counts[index[i]] += grad[i] * grad[i] * scale;
			}
		}
	}

	private void updateWeight(SparseFeatureVector fv) {
		double[] counts = featureSquaredCounts.get();
		double[] weights = weight.get();
		double[] weightDiffs = params.getUseAveraging() ? weightDiff.get() : null;
		for(VectorInfo vector:fv.getFeatureVectors()){
			int[] index = vector.getSv().getIndex();
			double[] grad = vector.getSv().getData();
			double scale = vector.getWeight();
			int size = vector.getSv().getUsed();
			for(int i = 0;i < size;i++){
				double d = eta / (delta + Math.sqrt(counts[index[i]]));
				double w = weights[index[i]];
				double next_w = w + d * grad[i] * scale; 
				//next_w = next_w > 0 ? Math.max(next_w - lambda, 0) : Math.min(next_w + lambda, 0);//L1?
				double diff = next_w - w;
				weights[index[i]] = next_w; 
				if(params.getUseAveraging()){
					weightDiffs[index[i]] += diff * trainStep;
				}
			}
		}
		trainStep++;
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
			// hinge loss
			double grad = (yStarState.getScore() + yStarState.getMargin()) > goldScore ? 1. : 0.;
			if(grad == 0.){
				continue;
			}
			SparseFeatureVector fvStar = fg.calculateFeature(instance, yStarState.getLabel(), yStarState.getLabel().size());
			SparseFeatureVector localFv = new SparseFeatureVector(params);
			localFv.add(grad, fvGold);
			localFv.add(-grad, fvStar);
			localFv.compact();
			updateVector.add(localFv);
			numUpdates++;
		}
		if(numUpdates > 0){
			updateVector.compact();	
			updateVector.scale(1. / params.getMiniBatch());
			updateSquaredCounts(updateVector);
			updateWeight(updateVector);
		}
	}

	public void save(String filename){
		try {
			BufferedWriter writer = new BufferedWriter(Files.newWriterSupplier(new File(filename), Charset.forName("UTF-8")).getOutput());
			writer.write(String.valueOf(trainStep));
			writer.write("\n");
			weight.save(writer);
			if(params.getUseAveraging()){
				weightDiff.save(writer);
				aveWeight.save(writer);
			}
			featureSquaredCounts.save(writer);
			params.saveModelParameters(writer);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void load(String filename){
		try {
			BufferedReader reader = new BufferedReader(Files.newReaderSupplier(new File(filename), Charset.forName("UTF-8")).getInput());
			trainStep = Integer.parseInt(reader.readLine().trim());
			weight.load(reader);
			if(params.getUseAveraging()){
				weightDiff.load(reader);
				aveWeight.load(reader);
			}
			featureSquaredCounts.load(reader);
			params.loadModelParameters(reader);
			reader.close();
			fg.init(); 		
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
