package data;

import java.util.Collection;
import java.util.Vector;

import config.Parameters;

public abstract class SequenceUnit{
	protected Parameters params;
	abstract public String getType();
	public abstract LabelUnit getNegativeClassLabel();

	public Collection<LabelUnit> getPossibleLabels() {
		Collection<LabelUnit> labels = params.getPossibleLabels(this.getType());
		if(labels == null){
			labels = new Vector<LabelUnit>();
			labels.add(getNegativeClassLabel());
		}else if(labels.size() == 0){
			labels.add(getNegativeClassLabel());
		}
		return labels;
	}
}
