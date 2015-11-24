package data.nlp.joint.pipeline;

import inference.State;

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Vector;
import java.util.concurrent.Callable;

import config.Parameters;
import data.Instance;
import data.Label;
import data.LabelUnit;
import data.nlp.joint.WordLabelUnit;
import model.SCWModel;
import model.SparseFeatureVector;
import model.Model;

public class PipelineState extends State {
	public PipelineState(Parameters params, Instance instance, boolean test) {
		super(params, instance, test);
	}
	public PipelineState(State state) {
		super(state);
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

	public PipelineState getNextGoldState(Model nerModel, Model relModel) {
		LabelUnit nextLabel = instance.getGoldLabel().getLabel(index+1);
		Model model = null;
		if(nextLabel instanceof WordLabelUnit){
			model = nerModel;
		}else{
			model = relModel;
		}
		SparseFeatureVector nextFv = model.getFeatureGenerator().calculateFeature(instance, label, index+1, nextLabel);
		// calculate update score 
		double localScore = model.evaluate(nextFv, false);
		// calculate next state
		PipelineState nextState = new PipelineState(this);
		nextState.update(model, localScore, nextFv, nextFv, nextLabel, true);
		assert nextState.index < nextState.instance.getSequence().size();
		assert margin == 0.;
		return nextState;
	}
	

	public Collection<Callable<Void>> getNextBestStatesTasks(final List<State> newBeamStates, final Model nerModel, final Model relModel, final int beamSize, final boolean test) {
		assert !params.getUseDynamicSort();
		List<Callable<Void>> tasks = new Vector<Callable<Void>>();
		Collection<LabelUnit> candidateLabels = instance.getCandidateLabels(this.label, index+1);
		final PipelineState currentState = this;
		final SparseFeatureVector goldFv = null;
		final LabelUnit goldLabel = instance.getGoldLabel().getLabel(index+1);
		assert !params.useSCW();
		for(final LabelUnit candidateLabel:candidateLabels){
			tasks.add(new Callable<Void>(){
				@Override
				public Void call() throws Exception {
					boolean correct = currentState.correct;
					PipelineState nextState = null;
					Model model = null;
					if(candidateLabel instanceof WordLabelUnit){
						model = nerModel;
					}else{
						model = relModel;
					}
					if(correct){
						if(candidateLabel.equals(goldLabel)){
							if(!params.getUseDynamicSort() && !test){
								assert !Double.isNaN(instance.getGoldScore(index+1));
								// gold ==> skip 
								nextState = new PipelineState(currentState);
								nextState.updateGold(goldLabel);
								assert Math.abs(nextState.getScore()-(currentState.getScore()+model.evaluate(model.getFeatureGenerator().calculateFeature(instance, label, index+1, candidateLabel), test))) < 1e-7;
							}	
						}else{
							correct = false;
						}	
					}
					if(nextState == null){
						nextState = new PipelineState(currentState);					
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
	
	public List<PipelineState> getNextBestStates(Model nerModel, Model relModel, int beamSize, boolean test) {
		SparseFeatureVector goldFv = null;
		assert !params.getUseDynamicSort();
		List<PipelineState> nextStates = new Vector<PipelineState>();
		Collection<LabelUnit> candidateLabels = instance.getCandidateLabels(this.label, index+1);
		LabelUnit goldLabel = instance.getGoldLabel().getLabel(index+1);
		assert !params.useSCW();
		for(LabelUnit candidateLabel:candidateLabels){
			assert candidateLabel.getClass().equals(goldLabel.getClass());
			boolean correct = this.correct;
			PipelineState nextState = null;
			Model model = null;
			if(candidateLabel instanceof WordLabelUnit){
				model = nerModel;
			}else{
				model = relModel;
			}			
			if(correct){
				if(candidateLabel.equals(goldLabel)){
					if(!params.getUseDynamicSort() && !test){
						assert !Double.isNaN(instance.getGoldScore(index+1));
						// gold ==> skip 
						nextState = new PipelineState(this);
						nextState.updateGold(goldLabel);
						assert Math.abs(nextState.getScore()-(this.getScore()+model.evaluate(model.getFeatureGenerator().calculateFeature(instance, label, index+1, candidateLabel), test))) < 1e-7;
					}
				}else{
					correct = false;
				}
			}
			if(nextState == null){
				nextState = new PipelineState(this);
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
