package data;

import java.util.List;
import java.util.Map;
import java.util.Vector;

import config.Parameters;

public abstract class Data {
	protected List<Instance> instances;
	protected Map<LabelUnit, Double> labelImportance;
	protected Parameters params;
	protected boolean isTrain;
	public Data(Parameters params, String fileBase, boolean isTrain){
		this.params = params;
		this.isTrain = isTrain;
		this.instances = new Vector<Instance>();
		load(fileBase);
		if(params.getVerbosity() > 2){
			System.out.println(fileBase+" loaded!!");
		}
	}
	public int size(){
		return instances.size();
	}
	public Instance getInstance(int idx){
		return instances.get(idx);
	}
	protected abstract void load(String dirBase);

	public double getLabelImportance(LabelUnit label){
		assert labelImportance.containsKey(label);
		assert params.getUseWeightedMargin()||params.getUseWeighting();
		return labelImportance.get(label);
	}
}
