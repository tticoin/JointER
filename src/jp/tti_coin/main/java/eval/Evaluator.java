package eval;

import java.util.List;

import config.Parameters;
import inference.Inference;
import inference.State;
import model.Model;
import data.Data;

public abstract class Evaluator {
	protected Parameters params;
	protected Inference infer;
	
	public Evaluator(Parameters params, Inference infer){
		this.params = params;
		this.infer = infer;
	}
	public void evaluate(Model model, Data valid){
		if(params.getVerbosity() == 0)return;
		int correct = 0;
		for(int idx = 0;idx < valid.size();idx++){
			List<State> yStarStates = getInference().inferBestState(model, valid.getInstance(idx), true);
			if(yStarStates.get(0).getLabel().equals(yStarStates.get(0).getGoldLabel())){
				correct++;
			}
		}
		System.out.println("Accuracy : "+ correct / (double)valid.size() + " ("+correct+"/"+valid.size()+")");
	}
	public Inference getInference() {
		return infer;
	}
	abstract public void predict(Model model, Data test);
}
