package model;

import java.util.List;
import java.util.Map.Entry;
import java.util.Vector;

import utils.HashToInt;

import no.uib.cipr.matrix.sparse.SparseVector;

import config.Parameters;

public class SparseFeatureVector {
	private Parameters params;
	private HashToInt h2i;
	private List<VectorInfo> featureVectors;

	public SparseFeatureVector(Parameters params) {
		this.params = params;
		this.h2i = new HashToInt(params);
		this.featureVectors = new Vector<VectorInfo>();
	}

	public void add(double weight, SparseFeatureVector vector) {
		for(VectorInfo featureVector:vector.featureVectors){
			if(featureVector.getSv().getUsed() > 0){
				featureVectors.add(new VectorInfo(featureVector.getKey(), featureVector.getWeight() * weight, featureVector.getSv()));
			}
		}
	}

	public void add(int key, double value) {
		featureVectors.get(0).getSv().add(key, value);
	}

	public void add(SparseFeatureVector vector) {
		for(VectorInfo featureVector:vector.featureVectors){
			if(featureVector.getSv().getUsed() > 0){
				featureVectors.add(new VectorInfo(featureVector));
			}
		}
	}

	public void add(SparseFeatureVector vector, int key) {
		for(VectorInfo featureVector:vector.featureVectors){
			if(featureVector.getSv().getUsed() > 0){
				featureVectors.add(new VectorInfo(key+featureVector.getKey(), featureVector.getWeight(), featureVector.getSv()));
			}
		}
	}

	public void add(SparseFeatureVector vector, String header) {
		assert vector != null;
		if(vector.size() == 0)return;
		add(vector, h2i.mapToInt(header));
	}

	public void add(StringSparseVector vector, String header) {
		if(vector.size() == 0)return;
		SparseVector featureVector = new SparseVector(params.fvSize());
		for(Entry<String, Double> entry:vector.getFv().entrySet()){
			featureVector.add(h2i.mapToInt(header.concat(entry.getKey())), entry.getValue());
		}
		featureVectors.add(new VectorInfo(0, 1., featureVector));
	}

	public void addToWeight(double w, WeightVector weight) {
		int filter = params.fvSize() - 1;
		double[] wArray = weight.get();
		for(VectorInfo featureVector:featureVectors){
			int size = featureVector.getSv().getUsed();
			int index = featureVector.getKey();
			int[] baseIndex = featureVector.getSv().getIndex();
			double[] value = featureVector.getSv().getData();
			double scale = featureVector.getWeight();
			if(index == 0){
				if(scale == 1.){
					for(int i = 0;i < size;i++){
						wArray[baseIndex[i]] += value[i] * w; 
					}
				}else{
					for(int i = 0;i < size;i++){
						wArray[baseIndex[i]] += value[i] * w * scale; 
					}					
				}
			}else{
				for(int i = 0;i < size;i++){
					wArray[(baseIndex[i]+index) & filter] += value[i] * w * scale; 
				}
			}
		}
	}
	

	public void compact() {
		if(featureVectors.size() == 1){
			return;
		}
		int filter = params.fvSize() - 1;
		SparseVector sv = new SparseVector(params.fvSize());
		for(VectorInfo featureVector:featureVectors){
			int size = featureVector.getSv().getUsed();
			int index = featureVector.getKey();
			int[] baseIndex = featureVector.getSv().getIndex();
			double[] value = featureVector.getSv().getData();
			double scale = featureVector.getWeight();
			for(int i = 0;i < size;i++){
				sv.add((baseIndex[i]+index) & filter, value[i]*scale);
				assert ((baseIndex[i]+index) & filter) == ((baseIndex[i]+index) % (filter+1)):baseIndex[i]+":"+index+":"+filter+":"+((baseIndex[i]+index) & filter)+":"+((baseIndex[i]+index) % (filter+1));
			}
		}
		this.featureVectors.clear();
		this.featureVectors.add(new VectorInfo(0, 1., sv));
	}
	
	public double dot(WeightVector weight) {
		double score = 0.;
		int filter = params.fvSize() - 1;
		double[] wArray = weight.get();
		for(VectorInfo featureVector:featureVectors){
			int size = featureVector.getSv().getUsed();
			int index = featureVector.getKey();
			int[] baseIndex = featureVector.getSv().getIndex();
			double[] value = featureVector.getSv().getData();
			double scale = featureVector.getWeight();
			if(index == 0){
				if(scale == 1.){
					for(int i = 0;i < size;i++){
						score += wArray[baseIndex[i]] * value[i];
					}
				}else{
					for(int i = 0;i < size;i++){
						score += wArray[baseIndex[i]] * value[i] * scale;
					}					
				}
			}else{
				for(int i = 0;i < size;i++){
					score += wArray[(baseIndex[i]+index) & filter] * value[i] * scale; 
				}
			}
		}
		return score;
	}

	public void extend() {
		featureVectors.add(new VectorInfo(0, 1., new SparseVector(params.fvSize())));		
	}

	public List<VectorInfo> getFeatureVectors() {
		return featureVectors;
	}

	public double getNorm(){
		double sum = 0.;
		for(VectorInfo featureVector:featureVectors){
			if(featureVector.getSv().getUsed() > 0){
				double localSum = 0.;
				for(double d:featureVector.getSv().getData()){
					localSum += d * d;
				}
				double weight = featureVector.getWeight();
				if(weight != 1.){
					localSum *= weight * weight;
				}
				sum += localSum;
			}
		}
		if(sum == 0.){
			return 0.;
		}
		return Math.sqrt(sum);
	}

	public SparseFeatureVector normalize() {
		return normalize(1.);
	}

	public SparseFeatureVector normalize(double f) {
		double norm = getNorm();
		if(norm != 0.){
			return scale(f/norm);
		}else{
			return this;
		}
	}

	public SparseFeatureVector scale(double d) {
		for(VectorInfo featureVector:featureVectors){
			featureVector.scale(d);
		}
		return this;
	}

	public int size() {
		int size = 0;
		for(VectorInfo featureVector:featureVectors){
			size += featureVector.getSv().getUsed();
		}
		return size;
	}

}
