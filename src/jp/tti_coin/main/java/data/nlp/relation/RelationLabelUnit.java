package data.nlp.relation;

import java.util.Collection;
import java.util.Set;
import java.util.TreeSet;
import java.util.Vector;

import config.Parameters;

import data.LabelUnit;
import data.SequenceUnit;

public class RelationLabelUnit implements LabelUnit {
	private String label;
	private boolean reverse;
	private boolean negative;

	public RelationLabelUnit(Parameters params, Collection<String> rels, EntityPair pair, String[] relArgs) {
		StringBuffer sb = new StringBuffer();
		if(rels.size() == 0){
			sb.append(Parameters.getNegativeClassLabel());
			sb.append("|");
			reverse = false;
			negative = true;
		}else if(params.getUseDirectedRelation()){
			assert relArgs.length == 2: relArgs;
			negative = false;
			int f = 0,r = 0;
			for(String rel:rels){
				if(rel.startsWith(relArgs[0])){
					f++;
				}else if(rel.startsWith(relArgs[1])){
					r++;
				}else{
					assert false;
				}
			}
			Set<String> normalisedRels = new TreeSet<String>();
			reverse = f < r;
			for(String rel:rels){
				if(reverse){
					if(rel.startsWith(relArgs[0])){
						rel = relArgs[1]+rel.substring(relArgs[0].length(), rel.length()-relArgs[1].length())+relArgs[0];
						normalisedRels.add(rel);
					}else if(rel.startsWith(relArgs[1])){
						rel = relArgs[0]+rel.substring(relArgs[1].length(), rel.length()-relArgs[0].length())+relArgs[1];
						normalisedRels.add(rel);
					}
				}else{
					normalisedRels.add(rel);
				}
			}
			for(String rel:normalisedRels){
				sb.append(rel);
				sb.append("|");
			}		
		}else{
			negative = false;
			reverse = false;
			for(String rel:rels){
				sb.append(rel);
				sb.append("|");
			}
		}
		
		if(params.getUseEntityTypedLabel()){	
			if(reverse){
				sb.append(pair.getE2().getType());
				sb.append("|");
				sb.append(pair.getE1().getType());
			}else{
				sb.append(pair.getE1().getType());
				sb.append("|");
				sb.append(pair.getE2().getType());
			}
			sb.append("|");
		}
		label = sb.toString().substring(0, sb.length()-1);
	}
	
	public RelationLabelUnit(String label, boolean reverse) {
		this.label = label;
		this.reverse = reverse;
		this.negative = label.startsWith(Parameters.getNegativeClassLabel());
	}

	public String getLabel() {
		return label;
	}

	public boolean isReverse() {
		return reverse;
	}

	@Override
	public boolean isNegative() {
		return negative;
	}

	@Override
	public boolean equals(Object unit) {
		if(!(unit instanceof RelationLabelUnit)){
			return false;
		}
		RelationLabelUnit labelUnit = (RelationLabelUnit)unit;
		if(this.isNegative() && labelUnit.isNegative()){
			return true;
		}
		if(getLabel().equals(labelUnit.getLabel())){
			return isReverse() == labelUnit.isReverse();
		}
		return false;
	}

	@Override
	public int compareTo(LabelUnit unit) {
		if(!(unit instanceof RelationLabelUnit)){
			return -1;
		}
		RelationLabelUnit labelUnit = (RelationLabelUnit)unit;
		if(isReverse() != labelUnit.isReverse()){
			if(isReverse()){
				return -1;
			}else{
				return 1;
			}
		}
		return getLabel().compareTo(labelUnit.getLabel());
	}

	@Override
	public String toString() {		
		//TODO: format
		return getLabel()+"$"+isReverse();
	}

	public static RelationLabelUnit getNegativeClassLabelUnit(Parameters params, SequenceUnit unit) {
		return new RelationLabelUnit(params, new Vector<String>(), (EntityPair)unit, null);
	}
	
}
