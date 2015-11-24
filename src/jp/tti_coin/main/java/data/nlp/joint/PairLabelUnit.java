package data.nlp.joint;

import java.util.Collection;
import java.util.Vector;

import config.Parameters;
import data.LabelUnit;

public class PairLabelUnit implements LabelUnit {
	String label;
	public PairLabelUnit(Collection<String> rels){
		StringBuffer sb = new StringBuffer();
		if(rels.size() == 0){
			sb.append(Parameters.getNegativeClassLabel());
			sb.append("|");
		}else{
			for(String rel:rels){
				sb.append(rel);
				sb.append("|");
			}
		}
		this.label = sb.toString();
	}
	
	public PairLabelUnit(String label) {
		this.label = label;
	}

	@Override
	public int compareTo(LabelUnit lu) {
		if(!(lu instanceof PairLabelUnit))return 1;
		return label.compareTo(((PairLabelUnit)lu).label);
	}

	@Override
	public boolean isNegative() {
		return label.startsWith(Parameters.getNegativeClassLabel());
	}
	
	public static LabelUnit getNegativeClassLabelUnit() {
		return new PairLabelUnit(new Vector<String>());
	}

	public String getType(){
		if(isNegative()){
			return Parameters.getNegativeClassLabel();
		}else{
			return label.split("\\|")[0].split(":")[1];
		}
	}
	
	public String getLabel(){
		return label;
	}
	
	@Override
	public String toString() {		
		return getLabel();
	}
	
	@Override
	public boolean equals(Object unit) {
		if(!(unit instanceof PairLabelUnit)){
			return false;
		}
		PairLabelUnit labelUnit = (PairLabelUnit)unit;
		if(this.isNegative() && labelUnit.isNegative()){
			return true;
		}
		return getLabel().equals(labelUnit.getLabel());
	}
}
