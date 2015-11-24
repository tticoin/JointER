package model;

import java.util.Map;
import java.util.Map.Entry;

import org.ardverk.collection.PatriciaTrie;
import org.ardverk.collection.StringKeyAnalyzer;

import config.Parameters;

public class StringSparseVector {
	protected Map<String, Double> vector;
	protected Parameters params;
	
	public StringSparseVector(Parameters params) {
		this.params = params;
		this.vector = new PatriciaTrie<String, Double>(StringKeyAnalyzer.CHAR);
	}
	
	public StringSparseVector(StringSparseVector vector) {
		this.params = vector.params;
		this.vector = new PatriciaTrie<String, Double>(StringKeyAnalyzer.CHAR);
		this.vector.putAll(vector.vector);
	}
	
	public void add(String key, double value){
		if(vector.containsKey(key)){
			vector.put(key, vector.get(key)+value);
		}else{
			vector.put(key, value);
		}
	}

	public void add(StringSparseVector additionalVector) {
		add(additionalVector, 1.);
	}

	public void add(StringSparseVector additionalVector, double weight) {
		for(Map.Entry<String, Double> entry:additionalVector.getFv().entrySet()){
			if(vector.containsKey(entry.getKey())){
				double val = vector.get(entry.getKey())+weight*entry.getValue();
				if(Math.abs(val) < Double.MIN_VALUE){
					vector.remove(entry.getKey());
				}else{
					vector.put(entry.getKey(), val);
				}
			}else{
				vector.put(entry.getKey(), weight*entry.getValue());
			}
		}
	}
	
	public void add(StringSparseVector vector, String header) {
		for(Entry<String, Double> entry:vector.vector.entrySet()){
			add(header.concat(entry.getKey()), entry.getValue());
		}		
	}

	public Map<String, Double> getFv() {
		return vector;
	}
	
	public double getNorm(){
		double sum = 0.;
		for(Map.Entry<String, Double> entry:vector.entrySet()){
			sum += entry.getValue() * entry.getValue();
		}
		sum = Math.sqrt(sum);
		return sum;
	}

	public void mult(double f) {
		for(Map.Entry<String, Double> entry:vector.entrySet()){
			entry.setValue(entry.getValue() * f);
		}	
	}

	public StringSparseVector normalize() {
		return normalize(1.);
	}

	public StringSparseVector normalize(double f) {
		double sum = 0.;
		for(Map.Entry<String, Double> entry:vector.entrySet()){
			sum += entry.getValue() * entry.getValue();
		}
		sum = Math.sqrt(sum);
		if(Math.abs(sum) < Double.MIN_VALUE){
			return this;
		}
		double isum = f / sum;
		for(Map.Entry<String, Double> entry:vector.entrySet()){
			entry.setValue(entry.getValue() * isum);
		}	
		return this;
	}

	public int size() {
		return getFv().size();
	}
}
