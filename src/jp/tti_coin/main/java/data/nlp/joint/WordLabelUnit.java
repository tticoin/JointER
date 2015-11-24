package data.nlp.joint;

import config.Parameters;
import data.LabelUnit;

public class WordLabelUnit implements LabelUnit {
	public static final String B = "B";
	public static final String I = "I";
	public static final String L = "L";
	public static final String O = "O";
	public static final String U = "U";

	private String position; 
	private String type; 
	
	public WordLabelUnit(String position, String type){
		this.position = position;
		this.type = type;
	}
	
	public WordLabelUnit(String value) {
		String[] attrs = value.split("-", 2);
		assert attrs.length == 2 && attrs[0].length() == 1;
		this.position = attrs[0];
		this.type = attrs[1];
	}

	@Override
	public int compareTo(LabelUnit lu) {
		if(!(lu instanceof WordLabelUnit))return -1;
		int comp = this.position.compareTo(((WordLabelUnit)lu).position);
		if(comp != 0)return comp;
		return type.compareTo(((WordLabelUnit)lu).type);
	}

	@Override
	public boolean isNegative() {
		return this.type.equals(Parameters.getNegativeClassLabel());
	}

	public String getPosition() {
		return position;
	}
	
	public String getType(){
		return type;
	}
	
	public String getLabel(){
		return position+"-"+type;
	}

	public static LabelUnit getNegativeClassLabelUnit() {
		return new WordLabelUnit(O, Parameters.getNegativeClassLabel());
	}
	
	@Override
	public String toString() {		
		return getLabel();
	}
	
	@Override
	public boolean equals(Object unit) {
		if(!(unit instanceof WordLabelUnit)){
			return false;
		}
		WordLabelUnit labelUnit = (WordLabelUnit)unit;
		if(this.isNegative() && labelUnit.isNegative()){
			return true;
		}
		return getLabel().equals(labelUnit.getLabel());
	}
}
