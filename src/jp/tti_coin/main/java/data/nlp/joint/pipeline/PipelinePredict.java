package data.nlp.joint.pipeline;

import model.FeatureGenerator;
import model.Model;
import config.Parameters;
import data.nlp.joint.JointData;

public class PipelinePredict {
	public static void main(String[] args){
		if(args.length != 1){
			System.err.println("Usage: java "+PipelinePredict.class.getName()+" parameters.yaml");
			System.exit(0);
		}
		Parameters params = Parameters.load(args[0]);
		JointData test = new JointData(params, params.getTestFolder(), true);
		PipelineInference infer = new PipelineInference(params);
		PipelineEvaluator evaluator = new PipelineEvaluator(params, infer);
		PipelineStructuredLearning sl = new PipelineStructuredLearning(params, infer, evaluator); 
		FeatureGenerator fg = new PipelineFeatureGenerator(params);
		Model nerModel = Model.create(params, fg);
		nerModel.load(params.getModelFile()+".0");
		Model relModel = Model.create(params, fg);
		relModel.load(params.getModelFile()+".1");
		sl.predict(nerModel, relModel, test);
	}
}
