package inference;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Vector;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import utils.MersenneTwister;
import config.Parameters;
import model.Model;
import data.Instance;

public class Inference {
	protected Parameters params;
	protected final Random rand;
	
	public Inference(Parameters params){
		this.params = params;
		this.rand = new MersenneTwister(0);
	}
	
	private List<State> getNextStatesParallel(List<State> beamStates, Model model, int beamSize, boolean test){
		List<Callable<Void>> tasks = new Vector<Callable<Void>>();
		List<State> newBeamStates = new Vector<State>();
		for(State currentState:beamStates){
			tasks.addAll(currentState.getNextBestStatesTasks(newBeamStates, model, beamSize, test));
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
	
	private List<State> getNextStates(List<State> beamStates, Model model, int beamSize, boolean test){
		List<State> newBeamStates = new Vector<State>();
		if(!test && rand.nextDouble() < params.getEpsilon()){
			for(State currentState:beamStates){
				assert currentState.hasNextState();
				// select next beam
				List<State> nextStates = currentState.getNextStatesRandom(model, beamSize, rand);
				newBeamStates.addAll(nextStates);
			}
			assert newBeamStates.size() > 0;
			if(newBeamStates.size() > beamSize){
				Collections.shuffle(newBeamStates, rand);
				newBeamStates = newBeamStates.subList(0, beamSize);
			}
			Collections.sort(newBeamStates);
		}else{
			if(params.getUseParallel()){
				return getNextStatesParallel(beamStates, model, beamSize, test);
			}
			for(State currentState:beamStates){
				assert currentState.hasNextState();
				// select next beam
				List<State> nextStates = currentState.getNextBestStates(model, beamSize, test);
				newBeamStates.addAll(nextStates);
				Collections.sort(newBeamStates);
				assert newBeamStates.size() < 2 || newBeamStates.get(0).getScore() + newBeamStates.get(0).getMargin() >= newBeamStates.get(1).getScore()+newBeamStates.get(1).getMargin():newBeamStates.get(0).getScore()+":"+newBeamStates.get(1).getScore();
				if(newBeamStates.size() > beamSize){
					newBeamStates = newBeamStates.subList(0, beamSize);
				}
				assert newBeamStates.size() <= beamSize;
			}
		}
		assert newBeamStates.size() <= beamSize;
		return newBeamStates;
	}
	
	public State findMaxViolatingState(Model model, Instance instance){
		if(params.useSort() && params.getUseGlobalFeatures()){
			instance.sort(model, false);
		}
		double[] scores = new double[instance.size()];
		if(params.getUseDynamicSort()){
			Arrays.fill(scores, Double.NaN);
		}else{
			State goldState = new State(params, instance, false);
			while(goldState.hasNextState()){
				goldState = goldState.getNextGoldState(model);
				scores[goldState.getIndex()] = goldState.getScore();
			}
		}
		instance.setGoldScores(scores);
		double maxViolatingScore = 0.;
		State maxViolatingState = null;
		int beamSize = params.getBeamSize();
		List<State> beamStates = new Vector<State>();
		beamStates.add(new State(params, instance, false));
		if(params.getUseBestFirst()){
			while(true){
				// select current best state 
				State currentState = beamStates.get(0);
				// no need to search next
				if(!currentState.hasNextState()){
					break;
				}
				assert currentState != null;
				// select next beam
				List<State> nextStates = getNextStates(beamStates.subList(0, 1), model, beamSize, false);
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
				beamStates = getNextStates(beamStates, model, beamSize, false);
				double violate = beamStates.get(0).getScore() + beamStates.get(0).getMargin() - beamStates.get(0).getInstance().getGoldScore(beamStates.get(0).getIndex());
				if(violate >= maxViolatingScore){
					maxViolatingState = beamStates.get(0);
					maxViolatingScore = violate;
				}
			}
			assert maxViolatingScore >= 0.:maxViolatingScore;
		}
		assert maxViolatingScore >= 0.:maxViolatingScore;
		//assert maxViolatingState != null;
		assert !beamStates.get(0).hasNextState();
		assert beamStates.get(0).getLabel().size() == instance.size() : beamStates.get(0).getLabel().size() +":"+ instance.size();
		instance.setGoldScores(null);
		return maxViolatingState;
	}

	public List<State> inferBestState(Model model, Instance instance, boolean test){
		if(params.useSort() && params.getUseGlobalFeatures()){
			instance.sort(model, test);	
		}
		if(!test){
			double[] scores = new double[instance.size()];
			if(params.getUseDynamicSort()){
				Arrays.fill(scores, Double.NaN);
			}else{
				State goldState = new State(params, instance, false);
				while(goldState.hasNextState()){
					goldState = goldState.getNextGoldState(model);
					scores[goldState.getIndex()] = goldState.getScore();
				}

			}
			instance.setGoldScores(scores);
		}
		int beamSize = params.getBeamSize();
		List<State> beamStates = new Vector<State>();	
		beamStates.add(new State(params, instance, test));
		if(params.getUseBestFirst()){
			while(true){
				// select current best state
				State currentState = beamStates.get(0);
				// no need to search next
				if(!currentState.hasNextState()){
					break;
				}
				assert currentState != null;
				// select next beam
				List<State> nextStates = getNextStates(beamStates.subList(0, 1), model, beamSize, test);
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
				beamStates = getNextStates(beamStates, model, beamSize, test);
				//System.err.println(beamStates.get(0).getLabel().getLabel(i).toString()+":"+instance.getGoldLabel().getLabel(i).toString());
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
