package data.nlp.joint.pipeline;

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
import learn.StructuredLearning;
import model.FeatureCache;
import model.FeatureGenerator;
import model.Model;
import data.Data;
import data.Instance;

public class PipelineStructuredLearning extends StructuredLearning {

	public PipelineStructuredLearning(Parameters params, PipelineInference infer, PipelineEvaluator evaluator) {
		super(params, infer, evaluator);
	}

	@Override
	protected Model createModel(FeatureGenerator fg) {
		return new PipelineDCDSSVMModel(params, fg);
	}

	public void predict(Model nerModel, Model relModel, Data test){
		// initialization
		long start = System.currentTimeMillis();	
		((PipelineEvaluator)evaluator).predict(nerModel, relModel, test);
		if(params.getVerbosity() > 2){
			System.out.format("Prediction finished in %d [msec] \n", System.currentTimeMillis() - start);
		}
	}
	
	public void learn(List<Model> models, Data data, FeatureGenerator fg, Data dev){
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
		

		Model nerModel = createModel(fg);
		Model relModel = createModel(fg);
		Random rand = new MersenneTwister(0);
		
		int size = data.size();

		assert !params.getUseLocalInit();
		// learning (NER)
		start = System.currentTimeMillis();
		int lastTrainStep = nerModel.getTrainStep();
		nerModel.averageWeight();
		for(int iter = 1;iter <= params.getIteration();iter++){
			if(params.useDCDSSVM()){
				for(int r = 0;r < 5;r++){
					Collections.shuffle(idxList, rand);	
					for(int idx:idxList){
						((PipelineDCDSSVMModel)nerModel).updateWeight(idx, true);
					}
				}
			}
			Collections.shuffle(idxList, rand);		
			for(int i = 0;i < size;i += params.getMiniBatch()){
				List<State> updates = new Vector<State>();
				for(int j = i;j < i + params.getMiniBatch() && j < size;j++){
					int idx = idxList.get(j);
					if(params.useMaxViolationUpdate()){
						updates.add(((PipelineInference)infer).findMaxViolatingState(nerModel, relModel, data.getInstance(idx), true));
					}else{				
						List<State> yStarStates = ((PipelineInference)infer).inferBestState(nerModel, relModel, data.getInstance(idx), false, true);
						updates.add(yStarStates.get(0));
					}
				}
				((PipelineDCDSSVMModel)nerModel).update(updates, true);
			}
			if(params.getVerbosity() > 1){
				if(iter % params.getOutputInterval() == 0){
					System.out.format("Iteration: %d, %d updates for %d instances in %d [msec]\n", iter, nerModel.getTrainStep() - lastTrainStep, idxList.size(), System.currentTimeMillis() - start);					
					// validation
					if(evaluator != null && dev != null){
						start = System.currentTimeMillis();
						nerModel.averageWeight();
						((PipelineEvaluator)evaluator).evaluate(nerModel, relModel, dev);
						System.out.format("Evaluation finished in %d [msec]\n", System.currentTimeMillis() - start);
					}
				}
			}
			if(nerModel.getTrainStep() == lastTrainStep){
				System.out.println("Converged!!");
				break;
			}
			lastTrainStep = nerModel.getTrainStep();
			start = System.currentTimeMillis();
		}
		// learning (NER)
		start = System.currentTimeMillis();
		lastTrainStep = relModel.getTrainStep();
		relModel.averageWeight();
		for(int iter = 1;iter <= params.getIteration();iter++){
			if(params.useDCDSSVM()){
				for(int r = 0;r < 5;r++){
					Collections.shuffle(idxList, rand);	
					for(int idx:idxList){
						((PipelineDCDSSVMModel)relModel).updateWeight(idx, true);
					}
				}
			}
			Collections.shuffle(idxList, rand);		
			for(int i = 0;i < size;i += params.getMiniBatch()){
				List<State> updates = new Vector<State>();
				for(int j = i;j < i + params.getMiniBatch() && j < size;j++){
					int idx = idxList.get(j);
					if(params.useMaxViolationUpdate()){
						updates.add(((PipelineInference)infer).findMaxViolatingState(nerModel, relModel, data.getInstance(idx), false));
					}else{				
						List<State> yStarStates = ((PipelineInference)infer).inferBestState(nerModel, relModel, data.getInstance(idx), false, false);
						updates.add(yStarStates.get(0));
					}
				}
				((PipelineDCDSSVMModel)relModel).update(updates, false);
			}
			if(params.getVerbosity() > 1){
				if(iter % params.getOutputInterval() == 0){
					System.out.format("Iteration: %d, %d updates for %d instances in %d [msec]\n", iter, relModel.getTrainStep() - lastTrainStep, idxList.size(), System.currentTimeMillis() - start);					
					// validation
					if(evaluator != null && dev != null){
						start = System.currentTimeMillis();
						relModel.averageWeight();
						((PipelineEvaluator)evaluator).evaluate(nerModel, relModel, dev);
						System.out.format("Evaluation finished in %d [msec]\n", System.currentTimeMillis() - start);
					}
				}
			}
			if(relModel.getTrainStep() == lastTrainStep){
				System.out.println("Converged!!");
				break;
			}
			lastTrainStep = relModel.getTrainStep();
			start = System.currentTimeMillis();
		}
		nerModel.averageWeight();
		relModel.averageWeight();
		//TODO: return both models
		models.add(nerModel);
		models.add(relModel);
	}
}
