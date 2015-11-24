package data.nlp.relation;

import learn.StructuredLearning;
import model.FeatureGenerator;
import model.Model;
import inference.Inference;
import config.Parameters;
import eval.Evaluator;

public class RelationPredict {
	public static void main(String[] args){
		if(args.length != 1){
			System.err.println("Usage: java "+RelationPredict.class.getName()+" parameters.yaml");
			System.exit(0);
		}
		Parameters params = Parameters.load(args[0]);
		RelationData test = new RelationData(params, params.getTestFolder(), true);
		Inference infer = new Inference(params);
		Evaluator evaluator = new RelationEvaluator(params, infer);
		StructuredLearning sl = new StructuredLearning(params, infer, evaluator); 
		FeatureGenerator fg = new RelationFeatureGenerator(params);
		Model model = Model.create(params, fg);
		model.load(params.getModelFile());
		sl.predict(model, test);
	}
}
