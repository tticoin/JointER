package data.nlp.joint;

import learn.StructuredLearning;
import model.FeatureGenerator;
import model.Model;
import inference.Inference;
import config.Parameters;
import eval.Evaluator;

public class JointPredict {
	public static void main(String[] args){
		if(args.length != 1){
			System.err.println("Usage: java "+JointPredict.class.getName()+" parameters.yaml");
			System.exit(0);
		}
		Parameters params = Parameters.load(args[0]);
		JointData test = new JointData(params, params.getTestFolder(), false);
		Inference infer = new Inference(params);
		Evaluator evaluator = new JointEvaluator(params, infer);
		StructuredLearning sl = new StructuredLearning(params, infer, evaluator); 
		FeatureGenerator fg = new JointFeatureGenerator(params);
		Model model = Model.create(params, fg);
		model.load(params.getModelFile());
		sl.predict(model, test);
	}
}
