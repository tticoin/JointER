package model;

import java.util.Map;
import java.util.Map.Entry;

import config.Parameters;

import data.nlp.Node;
import data.nlp.relation.RelationFeatureCache;
import data.nlp.relation.SimpleRelationFeatureCache;

public abstract class FeatureCache {
	protected SparseFeatureVector nodeFeatures = null;
	protected Map<Node, SparseFeatureVector> fullPathFeatures = null;
	protected Map<Node, SparseFeatureVector> shortestPathFeatures = null;
	protected Parameters params;
	protected Node node;
	
	public FeatureCache(Parameters params, Node node){
		this.params = params;
		this.node = node;
	}
	
	protected void add(Map<String, Float> from, Map<String, Float> to){
		for(Entry<String, Float> entry:from.entrySet()){
			add(entry.getKey(), entry.getValue(), to);
		}
	}
	
	protected void add(String key, float value, Map<String, Float> fv){
		if(fv.containsKey(key)){
			fv.put(key, fv.get(key)+value);
		}else{
			fv.put(key, value);
		}
	}
	
	public abstract void calcNodeFeatures();
	public abstract void calcPathFeatures(Node e2);

	public SparseFeatureVector getNodeFeatures(){
		assert nodeFeatures != null;
		return nodeFeatures;
	}
	
	public SparseFeatureVector getFullPathFeatures(Node e2){
		assert fullPathFeatures != null && fullPathFeatures.containsKey(e2);
		return fullPathFeatures.get(e2);
	}
	

	public SparseFeatureVector getShortestPathFeatures(Node e2){
		assert shortestPathFeatures != null && shortestPathFeatures.containsKey(e2);
		return shortestPathFeatures.get(e2);
	}
	//TODO
	public static void close() {
		RelationFeatureCache.close();		
		SimpleRelationFeatureCache.close();
	}
	
}
