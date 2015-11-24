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

import data.Data;
import data.Instance;

public abstract class Model {
	protected Parameters params;
	protected FeatureGenerator fg;
	protected WeightVector weight;
	protected WeightVector weightDiff;
	protected WeightVector aveWeight;
	protected int trainStep;
	
	public Model(Parameters params, FeatureGenerator fg){
		this.params = params;
		this.fg = fg;
		this.weight = new WeightVector(this.params);
		if(params.getUseAveraging()){
			this.weightDiff = new WeightVector(this.params);	
			this.aveWeight = new WeightVector(this.params);
		}
		this.trainStep = 1;
	}
	
	public void initialize(Data data) {
		for(int i = 0;i < data.size();i++){
			Instance instance = data.getInstance(i);
			SparseFeatureVector fvGold = fg.calculateFeature(instance, instance.getGoldLabel(), instance.getGoldLabel().size());
			fvGold.addToWeight(1./data.size(), weight);
			if(params.getUseAveraging()){
				fvGold.addToWeight(1./data.size(), weightDiff);
			}
			this.trainStep++;
		}
	}

	public void averageWeight(){
		if(params.getUseAveraging()){
			this.aveWeight = new WeightVector(this.params);
			aveWeight.add(weight);
			if(trainStep != 0){
				aveWeight.add(-1./trainStep, weightDiff);
			}
		}
	}
	
	public double evaluate(SparseFeatureVector sv, boolean average) {
		if(params.getUseAveraging() && average){
			assert aveWeight != null;
			return sv.dot(aveWeight);
		}else{
			return sv.dot(weight);
		}
	}

	public FeatureGenerator getFeatureGenerator() {
		return fg;
	}

	public int getTrainStep() {
		return trainStep;
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
			params.loadModelParameters(reader);
			reader.close();
			//TODO: refactor dependency of fg and model
			fg.init(); 		
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public abstract void update(List<State> updates);

	public void initialize(double w, double[] featureWeights) {
		double[] weights = this.weight.get();
		for(int i = 0;i < weights.length;i++){
			weights[i] = featureWeights[i] * w;
		}
	}
	
	public void initialize(Model model) {
		if(params.getUseAveraging()){
			this.weight.add(model.aveWeight);
		}else{
			this.weight.add(model.weight);
		}
	}

	public static Model create(Parameters params, FeatureGenerator fg) {
		Model model;
		if(params.usePerceptron()){
			model = new PerceptronModel(params, fg);
		}else if(params.useSGDSVM()){
			model = new SGDSVMModel(params, fg);
		}else if(params.useAdaGrad()){
			model = new AdaGradModel(params, fg);
		}else if(params.useDCDSSVM()){
			model = new DCDSSVMModel(params, fg);
		}else if(params.useAROW()){
			model = new AROWModel(params, fg);
		}else if(params.useSCW()){
			model = new SCWModel(params, fg);
		}else{
			model = null;
			assert false;
		}
		return model;
	}
	
}
