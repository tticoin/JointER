package data.nlp.joint;

import java.util.Collection;
import java.util.Set;
import java.util.TreeSet;

import config.Parameters;
import data.Label;
import data.LabelUnit;
import data.SequenceUnit;

public class Pair extends SequenceUnit {
	Parameters params;
	Word w1, w2;
	
	public Pair(Parameters params, Word w1, Word w2){
		this.params = params;
		this.w1 = w1;
		this.w2 = w2;
		assert this.w1.getWord().getOffset().compareTo(this.w2.getWord().getOffset()) <= 0;
	}

	@Override
	public String getType() {
		//TODO: check pair
		assert false;
		return null;
	}

	@Override
	public LabelUnit getNegativeClassLabel() {
		return PairLabelUnit.getNegativeClassLabelUnit();
	}

	public Word getW1() {
		return w1;
	}
	
	public Word getW2() {
		return w2;
	}
	
	public Collection<LabelUnit> getPossibleLabels() {
		assert false;
		return null;
	}
	
	public Collection<LabelUnit> getPossibleLabels(Label label, JointInstance instance) {
		int w1Idx = instance.getWord(w1.getId());
		String w1Type = "";
		int w2Idx = instance.getWord(w2.getId());
		String w2Type = "";
		if(params.getUseGoldEntitySpan()){
			String goldW1Position = ((WordLabelUnit)instance.getGoldLabel().getLabel(w1Idx)).getPosition();
			String goldW2Position = ((WordLabelUnit)instance.getGoldLabel().getLabel(w2Idx)).getPosition();
			if(!(goldW1Position.equals("U") || goldW1Position.equals("L")) ||
					!(goldW2Position.equals("U") || goldW2Position.equals("L"))){
				Set<LabelUnit> labels = new TreeSet<LabelUnit>();
				labels.add(getNegativeClassLabel());
				return labels;				
			}
			w1Type = instance.getGoldType(label, w1.getId());
			w2Type = instance.getGoldType(label, w2.getId());
		}
		if(w1Idx < label.size()){
			assert label.getLabel(w1Idx) instanceof WordLabelUnit;
			String position = ((WordLabelUnit)label.getLabel(w1Idx)).getPosition();
			if(position.equals(WordLabelUnit.B) || position.equals(WordLabelUnit.I) || position.equals(WordLabelUnit.O)){
				Set<LabelUnit> labels = new TreeSet<LabelUnit>();
				labels.add(getNegativeClassLabel());
				return labels;
			}			
			w1Type = ((WordLabelUnit)label.getLabel(w1Idx)).getType();
		}
		if(w1.getId() < instance.getNumWords() - 1){
			int nw1Idx = instance.getWord(w1.getId() + 1);
			if(nw1Idx < label.size()){
				String nextPosition = ((WordLabelUnit)label.getLabel(nw1Idx)).getPosition();
				if(nextPosition.equals(WordLabelUnit.I) || nextPosition.equals(WordLabelUnit.L)){
					Set<LabelUnit> labels = new TreeSet<LabelUnit>();
					labels.add(getNegativeClassLabel());
					return labels;				
				}
			}
		}		
		if(w1Type.equals("") && w1.getId() > 0){
			int pw1Idx = instance.getWord(w1.getId() - 1);
			if(pw1Idx < label.size()){
				WordLabelUnit unit = (WordLabelUnit)label.getLabel(pw1Idx);
				if(unit.getPosition().equals(WordLabelUnit.B)||unit.getPosition().equals(WordLabelUnit.I)){
					w1Type = unit.getType();
				}
			}
		}
		if(w2Idx < label.size()){
			assert label.getLabel(w2Idx) instanceof WordLabelUnit;
			String position = ((WordLabelUnit)label.getLabel(w2Idx)).getPosition();
			if(position.equals(WordLabelUnit.B) || position.equals(WordLabelUnit.I) || position.equals(WordLabelUnit.O)){
				Set<LabelUnit> labels = new TreeSet<LabelUnit>();
				labels.add(getNegativeClassLabel());
				return labels;
			}
			w2Type = ((WordLabelUnit)label.getLabel(w2Idx)).getType();
		}
		if(w2.getId() < instance.getNumWords() - 1){
			int nw2Idx = instance.getWord(w2.getId() + 1);
			if(nw2Idx < label.size()){
				String nextPosition = ((WordLabelUnit)label.getLabel(nw2Idx)).getPosition();
				if(nextPosition.equals(WordLabelUnit.I) || nextPosition.equals(WordLabelUnit.L)){
					Set<LabelUnit> labels = new TreeSet<LabelUnit>();
					labels.add(getNegativeClassLabel());
					return labels;				
				}
			}
		}
		if(w2Type.equals("") && w2.getId() > 0){
			int pw2Idx = instance.getWord(w2.getId() - 1);
			if(pw2Idx < label.size()){
				WordLabelUnit unit = (WordLabelUnit)label.getLabel(pw2Idx);
				if(unit.getPosition().equals(WordLabelUnit.B)||unit.getPosition().equals(WordLabelUnit.I)){
					w2Type = unit.getType();
				}
			}
		}
		String type = ":";
		if(params.getUseRelationTypeFilter()){
			type = w1Type+":"+w2Type;
		}
		if(!params.getPossibleLabels().containsKey(type)){
			Set<LabelUnit> labels = new TreeSet<LabelUnit>();
			labels.add(getNegativeClassLabel());
			return labels;
		}
				
		return params.getPossibleLabels(type);
	}

}
