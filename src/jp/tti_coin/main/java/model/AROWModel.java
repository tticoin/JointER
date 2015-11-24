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

// This is an implementaion of AROW.
// Avihai Mejer and Koby Crammer
// Confidence in Structured-Prediction using Confidence-Weighted Models. 
// EMNLP2010

public final class AROWModel extends Model {
	private WeightVector covariance;
	private final double r;
	private final double alpha = 1.;

	public AROWModel(Parameters params, FeatureGenerator fg) {
		super(params, fg);
		covariance = new WeightVector(params);
		covariance.fill(alpha);
		r = params.getLambda();
	}
	
	private double getConfidence(SparseFeatureVector fv){
		double c = 0.0;
		for(VectorInfo vector:fv.getFeatureVectors()){
			int[] index = vector.getSv().getIndex();
			double[] grad = vector.getSv().getData();
			double[] cov = covariance.get();
			int size = vector.getSv().getUsed();
			double localC = 0.;
			for(int i = 0;i < size;i++){				
				localC += cov[index[i]] * grad[i] * grad[i];
			}
			double scale = vector.getWeight();
			c += localC * scale * scale;
		}
		return c;
	}
	
	private void update(SparseFeatureVector fv, double alpha){
		double[] w = weight.get();
		double[] wdiff = null;
		if(params.getUseAveraging()){
			wdiff = weightDiff.get();
		}
		for(VectorInfo vector:fv.getFeatureVectors()){
			int[] index = vector.getSv().getIndex();
			double[] grad = vector.getSv().getData();
			double[] cov = covariance.get();
			double scale = vector.getWeight();
			int size = vector.getSv().getUsed();
			for(int i = 0;i < size;i++){
				int index_i = index[i];
				double cov_i = cov[index_i];
				double g = grad[i] * scale;
				double diff = alpha * cov_i * g;
				w[index_i] += diff;
				if(params.getUseAveraging()){
					wdiff[index_i] += trainStep * diff;
				}
				cov[index_i] = 1. / ((1. / cov_i) + g * g / r);
			}
		}
		trainStep++;
	}
	
	@Override
	public void update(List<State> updates) {
		SparseFeatureVector fv = new SparseFeatureVector(params);
		double gradSum = 0.;
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
			double grad = yStarState.getScore() + yStarState.getMargin() - goldScore;
			if(grad > 0.){
				SparseFeatureVector localFv = new SparseFeatureVector(params);
				localFv.add(1., fvGold);
				localFv.add(-1., fvStar);
				localFv.compact();
				fv.add(localFv);
				gradSum += grad;
			}
		}
		fv.compact();
		if(params.getMiniBatch() != 1){
			fv.scale(1./params.getMiniBatch());
			gradSum /= params.getMiniBatch();
		}
		if(gradSum != 0.){
			double alpha = gradSum / (r + getConfidence(fv)); 
			update(fv, alpha);
		}
	}
	

	@Override
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
			covariance.save(writer);
			params.saveModelParameters(writer);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	public void load(String filename){
		try {
			BufferedReader reader = new BufferedReader(Files.newReaderSupplier(new File(filename), Charset.forName("UTF-8")).getInput());
			trainStep = Integer.parseInt(reader.readLine().trim());
			weight.load(reader);
			if(params.getUseAveraging()){
				weightDiff.load(reader);
				aveWeight.load(reader);
			}
			covariance.load(reader);
			params.loadModelParameters(reader);
			reader.close();
			fg.init();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
