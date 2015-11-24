package data.nlp.joint.pipeline;

import model.SparseFeatureVector;
import config.Parameters;
import data.Instance;
import data.Label;
import data.nlp.joint.JointFeatureGenerator;
import data.nlp.joint.PairLabelUnit;
import data.nlp.joint.WordLabelUnit;

public class PipelineFeatureGenerator extends JointFeatureGenerator {
	
	public PipelineFeatureGenerator(Parameters params) {
		super(params);
	}

	public SparseFeatureVector calculateFeature(Instance instance, Label y, int size, boolean targetIsNE){
		SparseFeatureVector fv = new SparseFeatureVector(params);
		for(int i = 0;i < size;i++){
			if(targetIsNE && (y.getLabel(i) instanceof WordLabelUnit)){
				fv.add(calculateFeature(instance, y, i, y.getLabel(i)));
			}
			if(!targetIsNE && (y.getLabel(i) instanceof PairLabelUnit)){
				fv.add(calculateFeature(instance, y, i, y.getLabel(i)));
			}
		}
		return fv;
	}
	
}
