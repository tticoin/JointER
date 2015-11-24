package data.nlp.joint;

import learn.StructuredLearning;
import model.FeatureGenerator;
import model.Model;
import inference.Inference;
import config.Parameters;
import eval.Evaluator;

public class JointTrain {
	public static void main(String[] args){
		if(args.length != 1){
			System.err.println("Usage: java "+JointTrain.class.getName()+" parameters.yaml");
			System.exit(0);
		}
		Parameters params = Parameters.load(args[0]);
		JointData train = new JointData(params, params.getTrainFolder(), true);
		JointData dev = null;
		if(params.getDevFolder() != null){
			dev = new JointData(params, params.getDevFolder(), false);
		}
		Inference infer = new Inference(params);
		Evaluator evaluator = new JointEvaluator(params, infer);
		StructuredLearning sl = new StructuredLearning(params, infer, evaluator); 
		FeatureGenerator fg = new JointFeatureGenerator(params);
		Model model = sl.learn(train, fg, dev);
		if(model != null){
			model.save(params.getModelFile());
		}
	}
}
