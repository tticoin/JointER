package data;

import java.util.Iterator;
import java.util.List;
import java.util.Vector;

import config.Parameters;

public class Label {
	private Parameters params;
	private List<LabelUnit> labels;
	
	public Label(Parameters params){
		this.params = params;
		this.labels = new Vector<LabelUnit>();
	}
	
	public Label(Label label) {
		this.params = label.params;
		this.labels = new Vector<LabelUnit>(label.labels);
	}

	public void add(LabelUnit label) {
		labels.add(label);		
	}
	
	public int size(){
		return labels.size();
	}
	public LabelUnit getLabel(int index) {
		assert index >= 0 && index < labels.size(); 
		return labels.get(index);
	}
	
	public List<LabelUnit> getLabels() {
		return labels;
	}

	@Override
	public boolean equals(Object obj) {
		if(!(obj instanceof Label))return false;
		Label label = (Label)obj;
		if(this.size() != label.size()){
			return false;
		}
		Iterator<LabelUnit> thisIt = this.labels.iterator();
		Iterator<LabelUnit> labelIt = label.labels.iterator();
		while(thisIt.hasNext() && labelIt.hasNext()){
			if(!thisIt.next().equals(labelIt.next())){
				return false;
			}
		}
		return true;		
	}

}
