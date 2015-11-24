package learn;

import inference.Inference;
import inference.State;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Vector;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import com.google.common.io.Files;

import utils.MersenneTwister;

import config.Parameters;

import model.DCDSSVMModel;
import model.FeatureCache;
import model.FeatureGenerator;
import model.Model;

import data.Data;
import data.Instance;
import eval.Evaluator;

public class StructuredLearning {
	protected Parameters params;
	protected Inference infer;
	protected Evaluator evaluator;

	public StructuredLearning(Parameters params, Inference infer, Evaluator evaluator) {
		this.params = params;
		this.infer = infer;
		this.evaluator = evaluator;
		assert this.evaluator == null || this.evaluator.getInference() == infer;
	}

	public void predict(Model model, Data test){
		// initialization
		long start = System.currentTimeMillis();	
		evaluator.predict(model, test);
		if(params.getVerbosity() > 2){
			System.out.format("Prediction finished in %d [msec] \n", System.currentTimeMillis() - start);
		}
	}
	
	protected Model createModel(FeatureGenerator fg){
		return Model.create(params, fg);
	}
	
	public Model learn(Data data, FeatureGenerator fg, Data dev){
		// initialization
		long start = System.currentTimeMillis();	
		List<Integer> idxList = new Vector<Integer>();

		if(params.getUseParallel()){
			List<Callable<Void>> tasks = new Vector<Callable<Void>>();
			for(int idx = 0;idx < data.size();idx++){
				idxList.add(idx);
				final Instance instance = data.getInstance(idx);
				tasks.add(new Callable<Void>(){
					@Override
					public Void call() throws Exception {
						instance.cacheFeatures();
						return null;
					}
				});

			}
			if(dev != null){
				for(int idx = 0;idx < dev.size();idx++){
					final Instance instance = dev.getInstance(idx);
					tasks.add(new Callable<Void>(){
						@Override
						public Void call() throws Exception {
							instance.cacheFeatures();
							return null;
						}
					});
				}
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
		}else{
			if(params.getVerbosity() > 2){
				System.out.println("Training data size: "+data.size());
			}
			for(int idx = 0;idx < data.size();idx++){
				idxList.add(idx);
				data.getInstance(idx).cacheFeatures();
				if(params.getVerbosity() > 2){
					if((idx + 1) % 10 == 0){
						if((idx + 1) % 100 == 0){
							System.out.print("*");	
						}else{
							System.out.print(".");
						}
						System.out.flush();
					}
				}
			}
			if(params.getVerbosity() > 2){
				System.out.println();
			}
			if(dev != null){
				if(params.getVerbosity() > 2){
					System.out.println("Development data size: "+dev.size());
				}
				for(int idx = 0;idx < dev.size();idx++){
					dev.getInstance(idx).cacheFeatures();
					if((idx + 1) % 10 == 0){
						if((idx + 1) % 100 == 0){
							System.out.print("*");	
						}else{
							System.out.print(".");
						}
						System.out.flush();
					}
				}
				if(params.getVerbosity() > 2){
					System.out.println();
				}
			}
		}
		FeatureCache.close();
		if(params.getTrainVectorFile() != null){
			try{
				BufferedWriter writer = Files.newWriter(new File(params.getTrainVectorFile()), Charset.forName("US-ASCII"));
				for(int idx = 0;idx < data.size();idx++){
					Instance instance = data.getInstance(idx);
					fg.writeLocalFeatures(instance, writer);
				}
				writer.close();
			}catch(IOException ex){
				ex.printStackTrace();
			}
		}
		if(params.getDevVectorFile() != null && dev != null){
			try{
				BufferedWriter writer = Files.newWriter(new File(params.getDevVectorFile()), Charset.forName("US-ASCII"));
				for(int idx = 0;idx < dev.size();idx++){
					Instance instance = data.getInstance(idx);
					fg.writeLocalFeatures(instance, writer);
				}
				writer.close();
			}catch(IOException ex){
				ex.printStackTrace();
			}
		}
		if(params.getVerbosity() > 2){
			System.out.format("Feature calculation finished in %d [msec] \n", System.currentTimeMillis() - start);
			System.out.println("Start learning ...");
		}
		

		Model model = createModel(fg);
		Random rand = new MersenneTwister(0);

		int size = data.size();
		Model localModel = null;
		if(params.getUseLocalInit()){
			localModel = createModel(fg);
			assert params.getUseGlobalFeatures();
			long localInitStart = System.currentTimeMillis();
			boolean useDynamicSort = params.getUseDynamicSort(); 
			boolean useGlobal = params.getUseGlobalFeatures();
			params.setUseDynamicSort(false);
			params.setUseGlobalFeatures(false);
			for(int iter = 1;iter <= params.getLocalIteration();iter++){
				if(params.useDCDSSVM()){
					for(int r = 0;r < 5;r++){
						Collections.shuffle(idxList, rand);	
						for(int idx:idxList){
							((DCDSSVMModel)localModel).updateWeight(idx);
						}
					}
				}
				Collections.shuffle(idxList, rand);		
				for(int i = 0;i < size;i += params.getMiniBatch()){
					List<State> updates = new Vector<State>();
					for(int j = i;j < i + params.getMiniBatch() && j < size;j++){
						int idx = idxList.get(j);
						if(params.useMaxViolationUpdate()){
							State bestState = infer.findMaxViolatingState(localModel, data.getInstance(idx));
							if(bestState != null){
								updates.add(bestState);
							}
						}else{				
							List<State> yStarStates = infer.inferBestState(localModel, data.getInstance(idx), false);
							if(yStarStates.size() > 0){
								updates.add(yStarStates.get(0));
							}
						}
					}
					localModel.update(updates);
				}
			}
			localModel.averageWeight();
			model.initialize(localModel);
			model.averageWeight();
			params.setUseDynamicSort(useDynamicSort);
			params.setUseGlobalFeatures(useGlobal);
			// validation
			if(evaluator != null && dev != null){
				evaluator.evaluate(model, dev);
			}
			if(params.getVerbosity() > 2){
				System.out.format("Initialization finished in %d [msec] \n", System.currentTimeMillis() - localInitStart);
			}
		}
		
		// learning
		start = System.currentTimeMillis();
		int lastTrainStep = model.getTrainStep();
		model.averageWeight();
		for(int iter = 1;iter <= params.getIteration();iter++){
			if(params.useDCDSSVM()){
				for(int r = 0;r < 5;r++){
					Collections.shuffle(idxList, rand);	
					for(int idx:idxList){
						((DCDSSVMModel)model).updateWeight(idx);
					}
				}
			}
			Collections.shuffle(idxList, rand);		
			for(int i = 0;i < size;i += params.getMiniBatch()){
				List<State> updates = new Vector<State>();
				for(int j = i;j < i + params.getMiniBatch() && j < size;j++){
					int idx = idxList.get(j);
					if(params.useMaxViolationUpdate()){
						State bestState = infer.findMaxViolatingState(model, data.getInstance(idx));
						if(bestState != null){
							updates.add(bestState);
						}
					}else{				
						List<State> yStarStates = infer.inferBestState(model, data.getInstance(idx), false);
						if(yStarStates.size() > 0){
							updates.add(yStarStates.get(0));
						}
					}
				}
				model.update(updates);
			}
			if(params.getVerbosity() > 1){
				if(iter % params.getOutputInterval() == 0){
					System.out.format("Iteration: %d, %d updates for %d instances in %d [msec]\n", iter, model.getTrainStep() - lastTrainStep, idxList.size(), System.currentTimeMillis() - start);					
					// validation
					if(evaluator != null && dev != null){
						start = System.currentTimeMillis();
						model.averageWeight();
						evaluator.evaluate(model, dev);
						System.out.format("Evaluation finished in %d [msec]\n", System.currentTimeMillis() - start);
					}
				}
			}
			if(model.getTrainStep() == lastTrainStep){
				System.out.println("Converged!!");
				break;
			}
			if(localModel != null && iter == params.getLocalIteration()){
				localModel = null;
			}
			postIteration(model);
			lastTrainStep = model.getTrainStep();
			start = System.currentTimeMillis();
		}
		model.averageWeight();
		return model;
	}

	protected void postIteration(Model model) {
	}
}
