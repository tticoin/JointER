package data.nlp.joint.pipeline;

import java.util.List;
import java.util.Vector;

import model.FeatureGenerator;
import model.Model;
import config.Parameters;
import data.nlp.joint.JointData;

public class PipelineTrain {
	public static void main(String[] args){
		if(args.length != 1){
			System.err.println("Usage: java "+PipelineTrain.class.getName()+" parameters.yaml");
			System.exit(0);
		}
		Parameters params = Parameters.load(args[0]);
		JointData train = new JointData(params, params.getTrainFolder(), true);
		JointData dev = null;
		if(params.getDevFolder() != null){
			dev = new JointData(params, params.getDevFolder(), false);
		}
		PipelineInference infer = new PipelineInference(params);
		PipelineEvaluator evaluator = new PipelineEvaluator(params, infer);
		PipelineStructuredLearning sl = new PipelineStructuredLearning(params, infer, evaluator); 
		FeatureGenerator fg = new PipelineFeatureGenerator(params);
		List<Model> models = new Vector<Model>();
		sl.learn(models, train, fg, dev);
		if(models.size() != 0){
			for(int i = 0;i < models.size();i++){
				models.get(i).save(params.getModelFile()+"."+i);
			}
		}
	}
}
