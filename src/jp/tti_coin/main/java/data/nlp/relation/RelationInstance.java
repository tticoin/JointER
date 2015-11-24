package data.nlp.relation;

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Vector;

import com.google.common.collect.Multimap;
import com.google.common.collect.Ordering;
import com.google.common.collect.TreeMultimap;

import model.Model;
import config.Parameters;
import data.Instance;
import data.Label;
import data.LabelUnit;
import data.Sequence;
import data.nlp.Dijkstra;
import data.nlp.Node;
import data.nlp.Sentence;

public class RelationInstance extends Instance {
	private Sentence sentence;
	public RelationInstance(Parameters params, Sequence sequence,
			Label goldLabel, Sentence sentence) {
		super(params, sequence, goldLabel);
		this.sentence = sentence;
	}
	
	public RelationInstance(RelationInstance instance) {
		this(instance.params, instance.sequence, instance.goldLabel, instance.sentence);
		if(instance.goldScores != null){
			int length = instance.goldScores.length;
			goldScores = new double[length];
			System.arraycopy(instance.goldScores, 0, goldScores, 0, length);
		}
	}

	public Sentence getSentence() {
		return sentence;
	}

	@Override
	public void cacheFeatures() {
		for(int i = 0;i < size();i++){
			EntityPair pair = (EntityPair)sequence.get(i);
			if(pair.getE1().getCache() == null){
				if(params.getUseSimpleFeatures()){
					pair.getE1().setCache(new SimpleRelationFeatureCache(params, pair.getE1()));
				}else{
					pair.getE1().setCache(new RelationFeatureCache(params, pair.getE1()));
				}	
			}
			if(pair.getE2().getCache() == null){
				if(params.getUseSimpleFeatures()){
					pair.getE2().setCache(new SimpleRelationFeatureCache(params, pair.getE2()));
				}else{
					pair.getE2().setCache(new RelationFeatureCache(params, pair.getE2()));
				}
			}
			pair.getE1().getCache().calcNodeFeatures();
			pair.getE2().getCache().calcNodeFeatures();
			pair.getE1().getCache().calcPathFeatures(pair.getE2());
			pair.getE2().getCache().calcPathFeatures(pair.getE1());
		}
	}
	
	private class Distance implements Comparable<Distance> {
		private double score;
		private int structDist;
		private int charDist;
		private int offset;		
		
		public Distance(double score, int structDist, int charDist, int offset) {
			this.score = score;
			this.structDist = structDist;
			this.charDist = charDist;
			this.offset = offset;
		}

		@Override
		public int compareTo(Distance dist) {
			if(score != dist.score){
				return score < dist.score ? 1 : -1;//larger better
			}
			if(structDist != dist.structDist){
				return structDist - dist.structDist;//smaller better
			}
			if(charDist != dist.charDist){
				return charDist - dist.charDist;//smaller better
			}
			return dist.offset - offset; // larger better
		}
	}
	
	@Override
	public void sort() {
		// closest first
		Multimap<Distance, Integer> values = TreeMultimap.create(Ordering.natural(), Ordering.natural());
		Dijkstra dijkstra = new Dijkstra(params);
		for(int index = 0;index < size();index++){
			EntityPair pair = (EntityPair)sequence.get(index);
			int structDist = 0;
			if(params.getCloseAnnotation() != null){
				if(params.getParseAnnotationDescs().contains(params.getCloseAnnotation())){
					int minDistance = Integer.MAX_VALUE;
					for(Node e1Word:pair.getE1().getHeadWords(params.getSentenceAnnotation())){
						for(Node e2Word:pair.getE2().getHeadWords(params.getSentenceAnnotation())){
							List<List<Node>> shortestPaths = dijkstra.getShortestPath(e1Word, e2Word, params.getSentenceAnnotation());
							if(shortestPaths.size() != 0){
								int distance = shortestPaths.get(0).size();
								if(distance < minDistance){
									minDistance = distance;
								}
							}
						}
					}
					structDist = minDistance;
				}
			}
			int charDist = Math.min(
					Math.abs(pair.getE1().getOffset().getEnd() - pair.getE2().getOffset().getStart()), 
					Math.abs(pair.getE2().getOffset().getEnd() - pair.getE1().getOffset().getStart())
					);
			int offset = pair.getE1().getOffset().getEnd() + pair.getE2().getOffset().getEnd();
			Distance distance = new Distance(0., structDist, charDist, offset);
			values.put(distance, index);
		}
		Label label = new Label(params);
		Sequence sequence = new RelationSequence(params);
		assert size() == values.values().size();
		for(int index:values.values()){
			label.add(this.goldLabel.getLabel(index));
			sequence.add(this.sequence.get(index));
		}
		this.goldLabel = label;
		this.sequence = sequence;
	}

	@Override
	public void sort(Model model, Label y, int currentIndex, boolean test) {
		Dijkstra dijkstra = new Dijkstra(params);
		Multimap<Distance, Integer> values = null;
		if(params.useRandomFirst()){
			values = TreeMultimap.create(Ordering.arbitrary(), Ordering.natural());
		}else if(params.useEasyFirst()){
			values = TreeMultimap.create(Ordering.natural(), Ordering.natural());
		}else{
			assert params.useDifficultFirst();
			values = TreeMultimap.create(Ordering.natural().reverse(), Ordering.natural());
		}
		int target = currentIndex;
		for(;target < size();target++){
			EntityPair pair = (EntityPair)sequence.get(target);
			int structDist = 0;
			if(params.getCloseAnnotation() != null){
				if(params.getParseAnnotationDescs().contains(params.getCloseAnnotation())){
					int minDistance = Integer.MAX_VALUE;
					for(Node e1Word:pair.getE1().getHeadWords(params.getSentenceAnnotation())){
						for(Node e2Word:pair.getE2().getHeadWords(params.getSentenceAnnotation())){
							List<List<Node>> shortestPaths = dijkstra.getShortestPath(e1Word, e2Word, params.getSentenceAnnotation());
							if(shortestPaths.size() != 0){
								int distance = shortestPaths.get(0).size();
								if(distance < minDistance){
									minDistance = distance;
								}
							}
						}
					}
					structDist = minDistance;
				}
			}
			List<Double> scores = new Vector<Double>();
			Collection<LabelUnit> possibleLabels = pair.getPossibleLabels();
			int charDist = Math.min(
					Math.abs(pair.getE1().getOffset().getEnd() - pair.getE2().getOffset().getStart()), 
					Math.abs(pair.getE2().getOffset().getEnd() - pair.getE1().getOffset().getStart())
					);
			int offset = pair.getE1().getOffset().getEnd() + pair.getE2().getOffset().getEnd();
			if(possibleLabels.size() > 1){
				for(LabelUnit label:possibleLabels){
					double score = model.evaluate(model.getFeatureGenerator().calculateFeature(this, y, currentIndex - 1, target, label), test);
					scores.add(score);
				}

				int size = scores.size();
				Collections.sort(scores);
				Distance distance = new Distance(scores.get(size - 1), structDist, charDist, offset);
				values.put(distance, target);
			}else{
				Distance distance = new Distance(Double.POSITIVE_INFINITY, structDist, charDist, offset);
				values.put(distance, target);
			}
		}
		for(target++;target < size();target++){
			Distance distance = new Distance(Double.NEGATIVE_INFINITY, 0, 0, 0);
			values.put(distance, target);
		}
		Label label = new Label(params);
		Sequence sequence = new RelationSequence(params);
		assert values.values().size()+currentIndex == size();
		for(int index=0;index < currentIndex;index++){
			label.add(this.goldLabel.getLabel(index));
			sequence.add(this.sequence.get(index));
		}
		
		for(int index:values.values()){
			label.add(this.goldLabel.getLabel(index));
			sequence.add(this.sequence.get(index));
		}
		this.goldLabel = label;
		this.sequence = sequence;
	}
	
	
	@Override
	public void sort(Model model, boolean test) {
		Dijkstra dijkstra = new Dijkstra(params);
		Multimap<Distance, Integer> values = null;
		if(params.useRandomFirst()){
			values = TreeMultimap.create(Ordering.arbitrary(), Ordering.natural());
		}else if(params.useEasyFirst()){
			values = TreeMultimap.create(Ordering.natural(), Ordering.natural());
		}else{
			assert params.useDifficultFirst();
			values = TreeMultimap.create(Ordering.natural().reverse(), Ordering.natural());
		}
		for(int index = 0;index < size();index++){
			EntityPair pair = (EntityPair)sequence.get(index);
			int structDist = 0;
			if(params.getCloseAnnotation() != null){
				if(params.getParseAnnotationDescs().contains(params.getCloseAnnotation())){
					int minDistance = Integer.MAX_VALUE;
					for(Node e1Word:pair.getE1().getHeadWords(params.getSentenceAnnotation())){
						for(Node e2Word:pair.getE2().getHeadWords(params.getSentenceAnnotation())){
							List<List<Node>> shortestPaths = dijkstra.getShortestPath(e1Word, e2Word, params.getSentenceAnnotation());
							if(shortestPaths.size() != 0){
								int distance = shortestPaths.get(0).size();
								if(distance < minDistance){
									minDistance = distance;
								}
							}
						}
					}
					structDist = minDistance;
				}
			}
			List<Double> scores = new Vector<Double>();
			for(LabelUnit label:pair.getPossibleLabels()){
				double score = model.evaluate(model.getFeatureGenerator().calculateFeature(this, null, index, label, true), test);
				scores.add(score);
			}
			
			int charDist = Math.min(
					Math.abs(pair.getE1().getOffset().getEnd() - pair.getE2().getOffset().getStart()), 
					Math.abs(pair.getE2().getOffset().getEnd() - pair.getE1().getOffset().getStart())
					);
			int offset = pair.getE1().getOffset().getEnd() + pair.getE2().getOffset().getEnd();
			int size = scores.size();
			if(size > 1){
				Collections.sort(scores);
				Distance distance = new Distance(scores.get(size - 1), structDist, charDist, offset);
				values.put(distance, index);
			}else{
				Distance distance = new Distance(Double.POSITIVE_INFINITY, structDist, charDist, offset);
				values.put(distance, index);
			}
		}
		Label label = new Label(params);
		Sequence sequence = new RelationSequence(params);
		assert size() == values.values().size();
		for(int index:values.values()){
			label.add(this.goldLabel.getLabel(index));
			sequence.add(this.sequence.get(index));
		}
		this.goldLabel = label;
		this.sequence = sequence;
	}

	@Override
	public void clearCachedFeatures() {
		for(int i = 0;i < size();i++){
			EntityPair pair = (EntityPair)sequence.get(i);
			if(pair.getE1().getCache() != null){
				pair.getE1().setCache(null);
			}
			if(pair.getE2().getCache() != null){
				pair.getE2().setCache(null);
			}
		}		
	}
}
