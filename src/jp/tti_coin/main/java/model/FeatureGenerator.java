package model;

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;

import no.uib.cipr.matrix.VectorEntry;

import config.Parameters;
import data.Instance;
import data.Label;
import data.LabelUnit;

public abstract class FeatureGenerator {
	protected Parameters params;
	public FeatureGenerator(Parameters params){
		this.params = params;
		init();
	}
	public SparseFeatureVector calculateFeature(Instance instance, Label y, int size){
		SparseFeatureVector fv = new SparseFeatureVector(params);
		for(int i = 0;i < size;i++){
			fv.add(calculateFeature(instance, y, i, y.getLabel(i)));
		}
		return fv;
	}

	public SparseFeatureVector calculateFeature(Instance instance, Label y, int index, LabelUnit candidateLabel){
		return calculateFeature(instance, y, index, candidateLabel, false);
	}
	public SparseFeatureVector calculateFeature(Instance instance, Label y, int index, LabelUnit candidateLabel, boolean local){
		return calculateFeature(instance, y, index - 1, index, candidateLabel, local);
	}

	public SparseFeatureVector calculateFeature(Instance instance, Label y, int lastIndex, int target, LabelUnit candidateLabel){
		return calculateFeature(instance, y, lastIndex, target, candidateLabel, false);
	}
	public abstract SparseFeatureVector calculateFeature(Instance instance, Label y, int lastIndex, int target, LabelUnit candidateLabel, boolean local);
	
	public void writeLocalFeatures(Instance instance, BufferedWriter writer) throws IOException {
		boolean useGlobalFeatures = params.getUseGlobalFeatures();
		params.setUseGlobalFeatures(false);
		for(int i = 0;i < instance.size();i++){
			SparseFeatureVector localFv = calculateFeature(instance, instance.getGoldLabel(), i, instance.getSequence().get(i).getNegativeClassLabel());
			localFv.compact();
			LabelUnit label = instance.getGoldLabel().getLabel(i);
			writer.append(label.toString());
			Map<Integer, Double> fvMap = new TreeMap<Integer, Double>();
			for(VectorEntry entry:localFv.getFeatureVectors().get(0).getSv()){
				fvMap.put(entry.index(), entry.get());
			}
			for(Entry<Integer, Double> fvEntry:fvMap.entrySet()){
				writer.append(" ");
				writer.append(String.valueOf(fvEntry.getKey()));
				writer.append(":");
				writer.append(String.valueOf(fvEntry.getValue()));
			}
			writer.append("\n");
		}
		writer.flush();
		params.setUseGlobalFeatures(useGlobalFeatures);
	}
	
	public abstract void init();
	
}
