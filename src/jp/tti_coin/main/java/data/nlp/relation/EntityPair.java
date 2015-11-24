package data.nlp.relation;

import config.Parameters;
import data.LabelUnit;
import data.SequenceUnit;
import data.nlp.Node;

public class EntityPair extends SequenceUnit {
	private Node e1;
	private Node e2;
	
	public EntityPair(Parameters params, Node e1, Node e2) {
		this.params = params;
		this.e1 = e1;
		this.e2 = e2;
	}

	public Node getE1() {
		return e1;
	}

	public Node getE2() {
		return e2;
	}

	@Override
	public String getType() {
		return e1.getType()+":"+e2.getType();
	}

	@Override
	public LabelUnit getNegativeClassLabel() {
		return RelationLabelUnit.getNegativeClassLabelUnit(params, this);
	}
	

}
