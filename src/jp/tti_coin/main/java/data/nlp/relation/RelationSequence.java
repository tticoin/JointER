package data.nlp.relation;

import java.util.List;
import java.util.Vector;

import config.Parameters;

import data.Sequence;
import data.SequenceUnit;

public class RelationSequence implements Sequence {
	Parameters params;
	List<SequenceUnit> pairs;
	public RelationSequence(Parameters params){
		this.params = params;
		pairs = new Vector<SequenceUnit>();
	}
	
	@Override
	public void add(SequenceUnit unit){
		pairs.add(unit);
	}
		
	@Override
	public int size() {
		return pairs.size();
	}

	@Override
	public SequenceUnit get(int index) {
		assert index < pairs.size();
		return pairs.get(index);
	}

}
