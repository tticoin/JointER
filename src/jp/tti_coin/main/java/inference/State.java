package inference;

import java.lang.reflect.InvocationTargetException;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Vector;
import java.util.concurrent.Callable;

import config.Parameters;
import data.Instance;
import data.Label;
import data.LabelUnit;
import model.SCWModel;
import model.SparseFeatureVector;
import model.Model;

public class State implements Comparable<State> {
	protected Parameters params;
	protected Instance instance;
	protected Label label;
	protected double score;
	protected int index;
	protected double margin;
	protected boolean correct;
	protected boolean test;
	protected SparseFeatureVector diffFv;
	
	public State(Parameters params, Instance instance, boolean test) {
		this.params = params;
		if(params.getUseDynamicSort()){
			Class<?>[] types = {instance.getClass()};
			Object[] args = {instance};
			try {
				this.instance = instance.getClass().getConstructor(types).newInstance(args);
			} catch (InstantiationException | IllegalAccessException
					| IllegalArgumentException | InvocationTargetException
					| NoSuchMethodException | SecurityException e) {
				e.printStackTrace();
			}
		}else{
			this.instance = instance;
		}
		this.label = new Label(params);
		this.score = 0.;
		this.index = -1; 
		this.margin = 0.;
		this.test = test;
		this.correct = true;
		this.diffFv = null;
	}
	
	public State(State state) {
		this.params = state.params;
		if(params.getUseDynamicSort()){
			Class<?>[] types = {state.instance.getClass()};
			Object[] args = {state.instance};
			try {
				this.instance = state.instance.getClass().getConstructor(types).newInstance(args);
			} catch (InstantiationException | IllegalAccessException
					| IllegalArgumentException | InvocationTargetException
					| NoSuchMethodException | SecurityException e) {
				e.printStackTrace();
			}
		}else{
			this.instance = state.instance;
		}
		this.label = new Label(state.label);
		this.score = state.score;
		this.index = state.index;
		this.margin = state.margin;
		this.correct = state.correct;
		this.test = state.test;
		if(state.diffFv != null){
			this.diffFv = new SparseFeatureVector(params);
			this.diffFv.add(state.diffFv);
		}
	}
	public double getScore() {
		return score;
	}
	public int getIndex(){
		return index;
	}
	
	public double calculateMargin(Model model, LabelUnit candidateLabel, SparseFeatureVector goldFv, SparseFeatureVector nextFv){
		if(test){
			// test has no margin
			return margin;
		}
		if(params.useSCW()){
			// dynamic margin
			if(this.diffFv == null){
				diffFv = new SparseFeatureVector(params);
			}
			LabelUnit goldLabel = instance.getGoldLabel().getLabel(index+1);
			if((!params.getUseGlobalFeatures() || diffFv.size() == 0) && candidateLabel.equals(goldLabel)){
				// gold sequence or gold point with local features
				return margin;
			}
			SCWModel scwModel = (SCWModel)model;
			SparseFeatureVector diff = new SparseFeatureVector(params);
			diff.add(1., goldFv);
			diff.add(-1., nextFv);
			diff.compact();
			assert diff.size() > 0: goldLabel+":"+candidateLabel;
			diffFv.add(diff);
			diffFv.compact();
			return Math.sqrt(scwModel.getConfidence(diffFv)) * scwModel.getPhi();
		}else{
			LabelUnit goldLabel = this.instance.getGoldLabel().getLabel(this.index+1);
			if(candidateLabel.equals(goldLabel)){
				// Manhattan distance
				return margin;
			}else{
				if(params.getUseWeightedMargin()){
					return margin+params.getMargin()*this.instance.getData().getLabelImportance(goldLabel);
				}else{
					return margin+params.getMargin();
				}
			}
		}
	}

	public State getNextGoldState(Model model) {
		LabelUnit nextLabel = instance.getGoldLabel().getLabel(index+1);
		SparseFeatureVector nextFv = model.getFeatureGenerator().calculateFeature(instance, label, index+1, nextLabel);
		// calculate update score 
		double localScore = model.evaluate(nextFv, false);
		// calculate next state
		State nextState = new State(this);
		nextState.update(model, localScore, nextFv, nextFv, nextLabel, true);
		assert nextState.index < nextState.instance.getSequence().size();
		assert margin == 0.;
		return nextState;
	}
	

	public Collection<Callable<Void>> getNextBestStatesTasks(final List<State> newBeamStates, final Model model, final int beamSize, final boolean test) {
		SparseFeatureVector nextGoldFv = null;
		if(params.getUseDynamicSort()){
			instance.sort(model, this.label, index+1, test);
			if(!test){
				LabelUnit nextLabel = instance.getGoldLabel().getLabel(index+1);
				nextGoldFv = model.getFeatureGenerator().calculateFeature(instance, instance.getGoldLabel(), index+1, nextLabel);
				// calculate update score 
				double localGoldScore = model.evaluate(nextGoldFv, false);
				if(index == -1){
					instance.setGoldScore(index+1, localGoldScore);
				}else{
					instance.setGoldScore(index+1, instance.getGoldScore(index)+localGoldScore);
				}
			}
		}
		List<Callable<Void>> tasks = new Vector<Callable<Void>>();
		Collection<LabelUnit> candidateLabels = instance.getCandidateLabels(this.label, index+1);
		final State currentState = this;
		final SparseFeatureVector goldFv;
		final LabelUnit goldLabel = instance.getGoldLabel().getLabel(index+1);
		if(!test && params.useSCW()){
			if(params.getUseDynamicSort()){
				goldFv = nextGoldFv;
			}else{
				goldFv = model.getFeatureGenerator().calculateFeature(instance, instance.getGoldLabel(), index+1, goldLabel);
			}
		}else{
			goldFv = null;
		}
		for(final LabelUnit candidateLabel:candidateLabels){
			tasks.add(new Callable<Void>(){
				@Override
				public Void call() throws Exception {
					boolean correct = currentState.correct;
					State nextState = null;
					if(correct){
						if(candidateLabel.equals(goldLabel)){
							if(!params.getUseDynamicSort() && !test){
								assert !Double.isNaN(instance.getGoldScore(index+1));
								// gold ==> skip 
								nextState = new State(currentState);
								nextState.updateGold(goldLabel);
								assert Math.abs(nextState.getScore()-(currentState.getScore()+model.evaluate(model.getFeatureGenerator().calculateFeature(instance, label, index+1, candidateLabel), test))) < 1e-7;
							}	
						}else{
							correct = false;
						}	
					}
					if(nextState == null){
						nextState = new State(currentState);					
						SparseFeatureVector nextFv = model.getFeatureGenerator().calculateFeature(instance, label, index+1, candidateLabel);
						// calculate update score 
						double localScore = model.evaluate(nextFv, test);
						// calculate next state
						nextState.update(model, localScore, goldFv, nextFv, candidateLabel, correct);
					}
					newBeamStates.add(nextState);
					return null;
				}
			});
		}
		return tasks;	
	}
	

	public List<State> getNextStatesRandom(Model model, int beamSize, Random rand) {
		SparseFeatureVector goldFv = null;
		if(params.getUseDynamicSort()){
			instance.sort(model, this.label, index+1, false);
			LabelUnit nextLabel = instance.getGoldLabel().getLabel(index+1);
			goldFv = model.getFeatureGenerator().calculateFeature(instance, instance.getGoldLabel(), index+1, nextLabel);
			// calculate update score 
			double localGoldScore = model.evaluate(goldFv, false);
			if(index == -1){
				instance.setGoldScore(index+1, localGoldScore);
			}else{
				instance.setGoldScore(index+1, instance.getGoldScore(index)+localGoldScore);
			}
		}
		List<State> nextStates = new Vector<State>();
		Collection<LabelUnit> candidateLabels = instance.getCandidateLabels(this.label, index+1);
		LabelUnit goldLabel = instance.getGoldLabel().getLabel(index+1);
		if(params.useSCW() && goldFv == null){
			goldFv = model.getFeatureGenerator().calculateFeature(instance, instance.getGoldLabel(), index+1, goldLabel);
		}
		for(LabelUnit candidateLabel:candidateLabels){
			assert candidateLabel.getClass().equals(goldLabel.getClass());
			boolean correct = this.correct;
			State nextState = null;
			if(correct){
				if(candidateLabel.equals(goldLabel)){
					if(!params.getUseDynamicSort()){
						assert !Double.isNaN(instance.getGoldScore(index+1));
						// gold ==> skip 
						nextState = new State(this);
						nextState.updateGold(goldLabel);
						assert Math.abs(nextState.getScore()-(this.getScore()+model.evaluate(model.getFeatureGenerator().calculateFeature(instance, label, index+1, candidateLabel), false))) < 1e-7;
					}
				}else{
					correct = false;
				}
			}
			if(nextState == null){
				nextState = new State(this);
				SparseFeatureVector nextFv = model.getFeatureGenerator().calculateFeature(instance, label, index+1, candidateLabel);
				// calculate update score 
				double localScore = model.evaluate(nextFv, false);
				// calculate next state
				nextState.update(model, localScore, goldFv, nextFv, candidateLabel, correct);
				assert nextState.index < nextState.instance.getSequence().size();
			}
			assert nextState != null;
			nextStates.add(nextState);
		}
		if(nextStates.size() > beamSize){
			Collections.shuffle(nextStates, rand);
			nextStates = nextStates.subList(0, beamSize);
		}
		return nextStates;		
	}
	
	
	public List<State> getNextBestStates(Model model, int beamSize, boolean test) {
		SparseFeatureVector goldFv = null;
		if(params.getUseDynamicSort()){
			instance.sort(model, this.label, index+1, test);
			if(!test){
				LabelUnit nextLabel = instance.getGoldLabel().getLabel(index+1);
				goldFv = model.getFeatureGenerator().calculateFeature(instance, instance.getGoldLabel(), index+1, nextLabel);
				// calculate update score 
				double localGoldScore = model.evaluate(goldFv, false);
				if(index == -1){
					instance.setGoldScore(index+1, localGoldScore);
				}else{
					instance.setGoldScore(index+1, instance.getGoldScore(index)+localGoldScore);
				}
			}
		}
		List<State> nextStates = new Vector<State>();
		Collection<LabelUnit> candidateLabels = instance.getCandidateLabels(this.label, index+1);
		LabelUnit goldLabel = instance.getGoldLabel().getLabel(index+1);
		if(!test && params.useSCW() && goldFv == null){
			goldFv = model.getFeatureGenerator().calculateFeature(instance, instance.getGoldLabel(), index+1, goldLabel);
		}
		for(LabelUnit candidateLabel:candidateLabels){
			assert candidateLabel.getClass().equals(goldLabel.getClass());
			boolean correct = this.correct;
			State nextState = null;
			if(!test && correct){
				if(candidateLabel.equals(goldLabel)){
					if(!params.getUseDynamicSort() && !test){
						assert !Double.isNaN(instance.getGoldScore(index+1));
						// gold ==> skip 
						nextState = new State(this);
						nextState.updateGold(goldLabel);
						assert Math.abs(nextState.getScore()-(this.getScore()+model.evaluate(model.getFeatureGenerator().calculateFeature(instance, label, index+1, candidateLabel), test))) < 1e-7;
					}
				}else{
					correct = false;
				}
			}
			if(nextState == null){
				nextState = new State(this);
				SparseFeatureVector nextFv = model.getFeatureGenerator().calculateFeature(instance, label, index+1, candidateLabel);
				// calculate update score 
				double localScore = model.evaluate(nextFv, test);
				// calculate next state
				nextState.update(model, localScore, goldFv, nextFv, candidateLabel, correct);
				assert nextState.index < nextState.instance.getSequence().size();
			}
			nextStates.add(nextState);
			Collections.sort(nextStates);
			// store #beamSize states
			if(nextStates.size() >= beamSize){
				nextStates = nextStates.subList(0, beamSize);
			}
		}	
		assert nextStates.size() > 0;
		return nextStates;		
	}
	
	private void update(Model model, double localScore, SparseFeatureVector goldFv, SparseFeatureVector nextFv, LabelUnit candidateLabel, boolean correct) {
		this.score += localScore;
		this.margin = calculateMargin(model, candidateLabel, goldFv, nextFv);
		this.label.add(candidateLabel);
		this.correct = correct;
		this.index++;
		assert label.size() == index+1;
	}
	
	private void updateGold(LabelUnit goldLabel) {
		assert goldLabel.equals(this.instance.getGoldLabel().getLabel(index+1));
		assert this.correct;
		this.score = instance.getGoldScore(this.index+1);
		this.margin = 0.;
		this.label.add(goldLabel);
		this.index++;
		assert label.size() == index+1;
	}
		
	public boolean hasNextState() {
		return index+1 < instance.size();
	}
	
	public Label getLabel() {		
		return label;
	}
	@Override
	public int compareTo(State state) {
		//note that this includes margin!!!
		double thisScore = this.getScore()+this.getMargin();
		double stateScore = state.getScore()+state.getMargin();
		if(thisScore < stateScore){
			return 1;
		}else if(thisScore > stateScore){
			return -1;
		}
		return 0;
	}
	
	public double getMargin() {
		return margin;
	}

	public SparseFeatureVector getDiffFv() {
		return diffFv;
	}

	public boolean isCorrect() {
		return correct;
	}
	
	public Label getGoldLabel() {
		return instance.getGoldLabel();
	}

	public Instance getInstance() {
		return instance;
	}


}
