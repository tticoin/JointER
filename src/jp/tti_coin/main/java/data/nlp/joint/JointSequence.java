package data.nlp.joint;

import java.util.List;
import java.util.Vector;

import config.Parameters;
import data.Sequence;
import data.SequenceUnit;

public class JointSequence implements Sequence {
	Parameters params;
	List<SequenceUnit> pairs;
	public JointSequence(Parameters params){
		this.params = params;
		pairs = new Vector<SequenceUnit>();
	}
	
	public JointSequence(JointSequence sequence){
		this(sequence.params);
		pairs.addAll(sequence.pairs);
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
