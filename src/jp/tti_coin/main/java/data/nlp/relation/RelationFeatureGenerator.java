package data.nlp.relation;

import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import utils.HashToInt;

import model.FeatureGenerator;
import model.SparseFeatureVector;
import model.StringSparseVector;
import config.Parameters;
import data.Instance;
import data.Label;
import data.LabelUnit;
import data.nlp.Node;

public class RelationFeatureGenerator extends FeatureGenerator {
	private HashToInt h2i;
	//pre-calc values
	private Map<String, Integer> typeKeys;
	private final int GLOBAL_KEY;
	private final int PARA_KEY;
	//private final int PROJ_KEY;
	
	public RelationFeatureGenerator(Parameters params) {
		super(params);
		GLOBAL_KEY = h2i.mapToInt("Global");
		PARA_KEY = h2i.mapToInt("PARA");
		//PROJ_KEY = h2i.mapToInt("PROJPROJ");
	}
	
	@Override
	public void init() {
		h2i = new HashToInt(params);
		typeKeys = new TreeMap<String, Integer>();
		Set<LabelUnit> types = new TreeSet<LabelUnit>(params.getPossibleLabels().values());
		for(LabelUnit labelUnit:types){
			String type = ((RelationLabelUnit)labelUnit).getLabel();
			putToTypeKeys(type);
			putToTypeKeys("PROJ".concat(type));
			putToTypeKeys(type.concat("E1Node"));
			putToTypeKeys(type.concat("E2Node"));
			putToTypeKeys(type.concat("Path"));
			putToTypeKeys(type.concat("RPath"));
			for(LabelUnit labelUnit2:types){
				String type2 = ((RelationLabelUnit)labelUnit2).getLabel();
				putToTypeKeys("PROJ".concat(type).concat(type2));
			}
		}	
	}
	
	private void putToTypeKeys(String key){
		typeKeys.put(key, h2i.mapToInt(key));
	}

	@Override
	public SparseFeatureVector calculateFeature(Instance instance, Label y, int lastIndex, int target, LabelUnit candidateLabelUnit, boolean local) {
		SparseFeatureVector fv = new SparseFeatureVector(params);
		EntityPair pair = (EntityPair)instance.getSequence().get(target);
		String candidateLabel = ((RelationLabelUnit)candidateLabelUnit).getLabel();
		Node e1, e2;
		if(((RelationLabelUnit)candidateLabelUnit).isReverse()){
			e1 = pair.getE2();
			e2 = pair.getE1();			
		}else{
			e1 = pair.getE1();
			e2 = pair.getE2();
		}
		assert e1.getCache() != null && e2.getCache() != null;
		if(typeKeys.containsKey(candidateLabel.concat("Path"))){
			fv.add(e1.getCache().getFullPathFeatures(e2), typeKeys.get(candidateLabel.concat("Path")));
		}else{
			fv.add(e1.getCache().getFullPathFeatures(e2), h2i.mapToInt(candidateLabel.concat("Path")));
		}
		if(params.getUseGlobalFeatures() && !local){
			fv.add(calcGlobalFeatures(e1, e2, instance, y, lastIndex, (RelationLabelUnit)candidateLabelUnit), GLOBAL_KEY);
		}
		assert fv.size() > 0;
		fv.normalize(); 
		return fv;
	}
	
	protected void addSecondOrderInfoToFV(StringSparseVector fv, String label1, String base1, String label2, String head){
		fv.add(head.concat(label1).concat(base1).concat(label2), 1.);
		fv.add(head.concat(label1).concat(label2), 1.);
		fv.add(head.concat("1").concat(label1), 1.);
		fv.add(head.concat("2").concat(label2), 1.);
		fv.add(head, 1.);
	}
	
	protected SparseFeatureVector calcGlobalFeatures(Node e1, Node e2, Instance instance, Label y, int lastIndex, RelationLabelUnit candidateLabelUnit){
		SparseFeatureVector globalFv = new SparseFeatureVector(params);
		if(candidateLabelUnit.isNegative()){
			return globalFv;
		}
		String candidateLabel = candidateLabelUnit.getLabel();
		SparseFeatureVector parallelFv = new SparseFeatureVector(params);		
		StringSparseVector projectiveFv = new StringSparseVector(params);
		String parseAnnotationDesc = params.getSentenceAnnotation();
		String e1Base = e1.getFirstHeadWord(parseAnnotationDesc).getRealBase();
		String e2Base = e2.getFirstHeadWord(parseAnnotationDesc).getRealBase();	
		StringSparseVector counterFv = new StringSparseVector(params);
		for(int i = 0;i <= lastIndex;i++){
			RelationLabelUnit adjLabelUnit = (RelationLabelUnit)y.getLabel(i);
			if(adjLabelUnit.isNegative())continue;				
			String adjLabel = adjLabelUnit.getLabel();
			EntityPair adjPair = (EntityPair)instance.getSequence().get(i);
			Node adjE1, adjE2;
			if(adjLabelUnit.isReverse()){
				adjE1 = adjPair.getE2();
				adjE2 = adjPair.getE1();
			}else{
				adjE1 = adjPair.getE1();
				adjE2 = adjPair.getE2();
			}
			if(adjE1.equals(e1)){
				if(adjE2.getOffset().compareTo(e2.getOffset()) < 0){
					addSecondOrderInfoToFV(counterFv, candidateLabel, e1Base, adjLabel, "PARA-E1");
					//parallelFv.add(adjE2.getCache().getShortestPathFeatures(e2), "PARA-E1:".concat(candidateLabel).concat(adjLabel).concat(e1Base));
				}else{
					addSecondOrderInfoToFV(counterFv, adjLabel, e1Base, candidateLabel, "PARA-E1");
					//parallelFv.add(e2.getCache().getShortestPathFeatures(adjE2), "PARA-E1:".concat(adjLabel).concat(candidateLabel).concat(e1Base));
				}
			}else if(adjE2.equals(e1)){
				addSecondOrderInfoToFV(counterFv, candidateLabel, e1Base, adjLabel, "SEQ");
				//parallelFv.add(adjE1.getCache().getShortestPathFeatures(e2), "SEQ:".concat(candidateLabel).concat(adjLabel).concat(e1Base));
			}else if(adjE2.equals(e2)){
				if(adjE1.getOffset().compareTo(e1.getOffset()) < 0){
					addSecondOrderInfoToFV(counterFv, candidateLabel, e2Base, adjLabel, "PARA-E2");
					//parallelFv.add(adjE1.getCache().getShortestPathFeatures(e1), "PARA-E2:".concat(candidateLabel).concat(adjLabel).concat(e2Base));
				}else{
					addSecondOrderInfoToFV(counterFv, adjLabel, e2Base, candidateLabel, "PARA-E2");
					//parallelFv.add(e1.getCache().getShortestPathFeatures(adjE1), "PARA-E2:".concat(adjLabel).concat(candidateLabel).concat(e2Base));
				}
			}else if(adjE1.equals(e2)){
				addSecondOrderInfoToFV(counterFv, adjLabel, e2Base, candidateLabel, "SEQ");
				//parallelFv.add(e1.getCache().getShortestPathFeatures(adjE2), "SEQ:".concat(adjLabel).concat(candidateLabel).concat(e2Base));
			}else{
				int adj = adjE1.getOffset().getStart()+adjE2.getOffset().getStart();
				int target = e1.getOffset().getStart()+e2.getOffset().getStart();
				int adjMin = Math.min(adjE1.getOffset().getStart(), adjE2.getOffset().getStart());
				int adjMax = Math.max(adjE1.getOffset().getStart(), adjE2.getOffset().getStart());
				int targetMin = Math.min(e1.getOffset().getStart(), e2.getOffset().getStart());
				int targetMax = Math.max(e1.getOffset().getStart(), e2.getOffset().getStart());
				if(adj < target && adjMin < targetMin && targetMin < adjMax){
					addSecondOrderInfoToFV(projectiveFv, adjLabel, e2Base, candidateLabel, "PROJ");
				}else if(target < adj && targetMin < adjMin && adjMin < targetMax){
					addSecondOrderInfoToFV(projectiveFv, candidateLabel, e1Base, adjLabel, "PROJ");
				}
			}
		}
		globalFv.add(parallelFv.normalize(), PARA_KEY);
		globalFv.add(counterFv.normalize(), "COUNT");
		globalFv.add(projectiveFv.normalize(), "");
		return globalFv.normalize();		
	}

}
