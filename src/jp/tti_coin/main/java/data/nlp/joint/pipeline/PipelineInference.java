package data.nlp.joint.pipeline;

import inference.Inference;
import inference.State;

import java.util.Collections;
import java.util.List;
import java.util.Vector;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import config.Parameters;
import model.Model;
import data.Instance;
import data.nlp.joint.Pair;

public class PipelineInference extends Inference {
	public PipelineInference(Parameters params){
		super(params);
	}
	
	private List<State> getNextStatesParallel(List<State> beamStates, Model nerModel, Model relModel, int beamSize, boolean test){
		List<Callable<Void>> tasks = new Vector<Callable<Void>>();
		List<State> newBeamStates = new Vector<State>();
		for(State currentState:beamStates){
			tasks.addAll(((PipelineState)currentState).getNextBestStatesTasks(newBeamStates, nerModel, relModel, beamSize, test));
		}		
		try {
			ExecutorService threadPool;
			if(params.getProcessors() > 0){
				threadPool = Executors.newFixedThreadPool(params.getProcessors());
			}else{
				threadPool = Executors.newCachedThreadPool();
			}
			threadPool.invokeAll(tasks);
			threadPool.shutdown();
			threadPool.awaitTermination(Long.MAX_VALUE, TimeUnit.SECONDS);		
		} catch (InterruptedException e) {
			throw new RuntimeException(e);
		} 
		Collections.sort(newBeamStates);
		assert newBeamStates.size() < 2 || newBeamStates.get(0).getScore() + newBeamStates.get(0).getMargin() >= newBeamStates.get(1).getScore()+newBeamStates.get(1).getMargin():newBeamStates.get(0).getScore()+":"+newBeamStates.get(1).getScore();
		if(newBeamStates.size() > beamSize){
			newBeamStates = newBeamStates.subList(0, beamSize);
		}
		assert newBeamStates.size() <= beamSize;
		return newBeamStates;
	}
	
	private List<State> getNextStates(List<State> beamStates, Model nerModel, Model relModel, int beamSize, boolean test){
		if(params.getUseParallel()){
			return getNextStatesParallel(beamStates, nerModel, relModel, beamSize, test);
		}
		List<State> newBeamStates = new Vector<State>();
		for(State currentState:beamStates){
			assert currentState.hasNextState();
			// select next beam
			List<PipelineState> nextStates = ((PipelineState)currentState).getNextBestStates(nerModel, relModel, beamSize, test);
			newBeamStates.addAll(nextStates);
			Collections.sort(newBeamStates);
			assert newBeamStates.size() < 2 || newBeamStates.get(0).getScore() + newBeamStates.get(0).getMargin() >= newBeamStates.get(1).getScore()+newBeamStates.get(1).getMargin():newBeamStates.get(0).getScore()+":"+newBeamStates.get(1).getScore();
			if(newBeamStates.size() > beamSize){
				newBeamStates = newBeamStates.subList(0, beamSize);
			}
			assert newBeamStates.size() <= beamSize;
		}
		assert newBeamStates.size() <= beamSize;
		return newBeamStates;
	}
	
	public State findMaxViolatingState(Model nerModel, Model relModel, Instance instance, boolean neOnly){
		assert !params.useSort() && !params.getUseDynamicSort();
		double[] scores = new double[instance.size()];
		PipelineState goldState = new PipelineState(params, instance, false);
		while(goldState.hasNextState()){
			goldState = goldState.getNextGoldState(nerModel, relModel);
			scores[goldState.getIndex()] = goldState.getScore();
		}
		instance.setGoldScores(scores);
		double maxViolatingScore = 0.;
		State maxViolatingState = null;
		int beamSize = params.getBeamSize();
		List<State> beamStates = new Vector<State>();
		beamStates.add(new PipelineState(params, instance, false));
		if(params.getUseBestFirst()){
			while(true){
				// select current best state 
				State currentState = beamStates.get(0);
				// no need to search next
				if(!currentState.hasNextState()){
					break;
				}
				if(neOnly){
					int index = currentState.getIndex();
					if(currentState.getInstance().getSequence().get(index+1) instanceof Pair){
						break;
					}
				}				
				assert currentState != null;
				// select next beam
				List<State> nextStates = getNextStates(beamStates.subList(0, 1), nerModel, relModel, beamSize, false);
				beamStates.remove(currentState);
				beamStates.addAll(nextStates);
				Collections.sort(beamStates);
				assert beamStates.size() < 2 || beamStates.get(0).getScore() + beamStates.get(0).getMargin() >= beamStates.get(1).getScore()+beamStates.get(1).getMargin();
				if(beamStates.size() > beamSize){
					beamStates = beamStates.subList(0, beamSize);
				}
				double violate = beamStates.get(0).getScore() + beamStates.get(0).getMargin() - beamStates.get(0).getInstance().getGoldScore(beamStates.get(0).getIndex());
				if(violate >= maxViolatingScore){
					maxViolatingState = beamStates.get(0);
					maxViolatingScore = violate;
					assert violate >= 0.:violate;
				}
			}
		}else{
			int size = instance.size();
			for(int i = 0;i < size;i++){
				if(neOnly){
					if(beamStates.get(0).getInstance().getSequence().get(i) instanceof Pair){
						break;
					}
				}		
				beamStates = getNextStates(beamStates, nerModel, relModel, beamSize, false);
				double violate = beamStates.get(0).getScore() + beamStates.get(0).getMargin() - beamStates.get(0).getInstance().getGoldScore(beamStates.get(0).getIndex());
				if(violate >= maxViolatingScore){
					maxViolatingState = beamStates.get(0);
					maxViolatingScore = violate;
				}
			}
			assert maxViolatingScore >= 0.:maxViolatingScore;
		}
		assert maxViolatingScore >= 0.:maxViolatingScore;
		assert maxViolatingState != null;
		assert !beamStates.get(0).hasNextState();
		assert beamStates.get(0).getLabel().size() == instance.size() : beamStates.get(0).getLabel().size() +":"+ instance.size();
		instance.setGoldScores(null);
		return maxViolatingState;
	}

	public List<State> inferBestState(Model nerModel, Model relModel, Instance instance, boolean test, boolean neOnly){
		assert !params.useSort() && !params.getUseDynamicSort();
		if(!test){
			double[] scores = new double[instance.size()];
			PipelineState goldState = new PipelineState(params, instance, false);
			while(goldState.hasNextState()){
				goldState = goldState.getNextGoldState(nerModel, relModel);
				scores[goldState.getIndex()] = goldState.getScore();
			}
			instance.setGoldScores(scores);
		}
		int beamSize = params.getBeamSize();
		List<State> beamStates = new Vector<State>();	
		beamStates.add(new PipelineState(params, instance, test));
		if(params.getUseBestFirst()){
			while(true){
				// select current best state
				State currentState = beamStates.get(0);
				// no need to search next
				if(!currentState.hasNextState()){
					break;
				}
				if(neOnly){
					int index = currentState.getIndex();
					if(currentState.getInstance().getSequence().get(index+1) instanceof Pair){
						break;
					}
				}			
				assert currentState != null;
				// select next beam
				List<State> nextStates = getNextStates(beamStates.subList(0, 1), nerModel, relModel, beamSize, test);
				beamStates.remove(currentState);
				beamStates.addAll(nextStates);
				Collections.sort(beamStates);
				assert beamStates.size() < 2 || beamStates.get(0).getScore() + beamStates.get(0).getMargin() >= beamStates.get(1).getScore()+beamStates.get(1).getMargin();
				if(beamStates.size() > beamSize){
					beamStates = beamStates.subList(0, beamSize);
				}
				// check early update
				if(params.useEarlyUpdate() && !test){
					boolean earlyUpdate = true;
					for(State state:beamStates){
						if(state.isCorrect()){
							earlyUpdate = false;
						}
					}
					if(earlyUpdate){
						break;
					}
				}
			}
		}else{
			int size = instance.size();
			for(int i = 0;i < size;i++){
				if(neOnly){
					if(beamStates.get(0).getInstance().getSequence().get(i) instanceof Pair){
						break;
					}
				}		
				beamStates = getNextStates(beamStates, nerModel, relModel, beamSize, test);
				// check early update
				if(params.useEarlyUpdate() && !test){
					boolean earlyUpdate = true;
					for(State state:beamStates){
						if(state.isCorrect()){
							earlyUpdate = false;
						}
					}
					if(earlyUpdate){
						break;
					}
				}
				assert beamStates.get(0).getIndex() == i;
			}
		}
		assert (params.useEarlyUpdate() && !test) || !beamStates.get(0).hasNextState();
		assert (params.useEarlyUpdate() && !test) || beamStates.get(0).getLabel().size() == instance.size() : beamStates.get(0).getLabel().size() +":"+ instance.size();
		instance.setGoldScores(null);
		return beamStates;
	}
}
