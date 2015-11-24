package data;

import java.util.Collection;

import model.Model;
import config.Parameters;

public abstract class Instance {
	protected Parameters params;
	protected Label goldLabel;
	protected Sequence sequence;
	protected int index; 
	protected Data data;
	protected double[] goldScores;
	public Instance(Parameters params, Sequence sequence, Label goldLabel){
		assert sequence.size() == goldLabel.size();
		this.params = params;
		this.sequence = sequence;
		this.goldLabel = goldLabel;
		this.goldScores = null;
	}	
	public Instance(Instance instance){
		this(instance.params, instance.sequence, instance.goldLabel);
	}
	public Label getGoldLabel(){
		return goldLabel;
	}
	public Sequence getSequence() {
		return sequence;
	}
	public int size() {
		return sequence.size();
	}
	public int getIndex() {
		return index;
	}
	public void setIndex(int index) {
		this.index = index;
	}
	public Collection<LabelUnit> getCandidateLabels(Label label, int index) {
		assert index < goldLabel.size();
		assert sequence.get(index).getPossibleLabels().size() > 0;
		return sequence.get(index).getPossibleLabels();
	}
	public abstract void sort(Model model, boolean test);
	public abstract void sort(Model model, Label y, int currentIndex, boolean test);
	public abstract void cacheFeatures();
	public abstract void clearCachedFeatures();
	public abstract void sort();
	public void setData(Data data) {
		this.data = data;
	}
	public Data getData(){
		return data;
	}
	public void setGoldScores(double[] goldScores) {
		this.goldScores = goldScores;
	}
	public void setGoldScore(int index, double value) {
		goldScores[index] = value;
	}

	public double getGoldScore(int index) {
		return goldScores[index];
	}
}
