package model;

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;

import no.uib.cipr.matrix.VectorEntry;

import config.Parameters;
import data.Data;
import data.Instance;
import data.Label;
import data.LabelUnit;
import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Problem;

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
	
	public Problem buildProblem(Data data){
		boolean useGlobalFeatures = params.getUseGlobalFeatures();
		params.setUseGlobalFeatures(false);
		Problem problem = new Problem();
		int size = 0;
		for(int i = 0;i < data.size();i++){
			Instance instance = data.getInstance(i);
			for(int j = 0;j < instance.size();j++){
				size += instance.getCandidateLabels(instance.getGoldLabel(), j).size();
			}
		}	
		problem.l = size;
		problem.n = params.fvSize() + 1;
		problem.y = new double[problem.l];
		problem.x = new Feature[problem.l][];
		problem.labels = new LabelUnit[problem.l];
		problem.bias = -1;
		int idx = 0;
		for(int i = 0;i < data.size();i++){
			Instance instance = data.getInstance(i);
			for(int j = 0;j < instance.size();j++){
				for(LabelUnit candidateLabel:instance.getCandidateLabels(instance.getGoldLabel(), j)){
					SparseFeatureVector localFv = calculateFeature(instance, instance.getGoldLabel(), j, candidateLabel);
					localFv.compact();
					if(candidateLabel.equals(instance.getGoldLabel().getLabel(j))){
						problem.y[idx] = 1;
					}else{
						problem.y[idx] = -1;
					}
					problem.labels[idx] = candidateLabel;
					Map<Integer, Double> fvMap = new TreeMap<Integer, Double>();
					for(VectorEntry entry:localFv.getFeatureVectors().get(0).getSv()){
						assert !fvMap.containsKey(entry.index());
						fvMap.put(entry.index(), entry.get());
					}
					problem.x[idx] = new Feature[fvMap.size()];
					int fvIdx = 0;
					int lastIndex = -1;
					for(Entry<Integer, Double> fvEntry:fvMap.entrySet()){
						assert fvEntry.getKey() >= 0 && fvEntry.getKey() < params.fvSize(): fvEntry.getKey();
						assert lastIndex < fvEntry.getKey(): lastIndex +":"+fvEntry.getKey();
						//NOTE: index > 0
						problem.x[idx][fvIdx] = new FeatureNode(fvEntry.getKey()+1, fvEntry.getValue().floatValue());						
						fvIdx++;
						lastIndex = fvEntry.getKey();
					}
					idx++;
				}
			}
		}
		params.setUseGlobalFeatures(useGlobalFeatures);
		return problem;
	}
	public abstract void init();
	
}
