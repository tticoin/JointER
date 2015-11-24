package data.nlp.relation;

import learn.StructuredLearning;
import model.FeatureGenerator;
import model.Model;
import inference.Inference;
import config.Parameters;
import eval.Evaluator;

public class RelationTrain {
	public static void main(String[] args){
		if(args.length != 1){
			System.err.println("Usage: java "+RelationTrain.class.getName()+" parameters.yaml");
			System.exit(0);
		}
		Parameters params = Parameters.load(args[0]);
		RelationData train = new RelationData(params, params.getTrainFolder(), true);
		RelationData dev = null;
		if(params.getDevFolder() != null){
			dev = new RelationData(params, params.getDevFolder(), false);
		}
		Inference infer = new Inference(params);
		Evaluator evaluator = new RelationEvaluator(params, infer);
		StructuredLearning sl = new StructuredLearning(params, infer, evaluator); 
		FeatureGenerator fg = new RelationFeatureGenerator(params);
		Model model = sl.learn(train, fg, dev);
		if(model != null){
			model.save(params.getModelFile());
		}
	}
}
