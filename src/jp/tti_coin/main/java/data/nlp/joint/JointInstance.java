package data.nlp.joint;

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Vector;

import com.google.common.collect.Multimap;
import com.google.common.collect.Ordering;
import com.google.common.collect.TreeMultimap;

import config.Parameters;
import model.Model;
import data.Instance;
import data.Label;
import data.LabelUnit;
import data.Sequence;
import data.SequenceUnit;
import data.nlp.Node;

public class JointInstance extends Instance {
	private int[] words;
	private int[][] wordPairs;
	
	public JointInstance(Parameters params, Sequence sequence, Label goldLabel) {
		super(params, sequence, goldLabel);
		int nwords = 0;
		int size = size();
		for(int index = 0;index < size;++index){
			if(sequence.get(index) instanceof Word){
				nwords++;
			}
		}
		words = new int[nwords];
		wordPairs = new int[nwords][nwords];
		for(int index = 0;index < size;++index){
			SequenceUnit unit = sequence.get(index);
			if(unit instanceof Word){
				words[((Word)unit).getId()] = index;
			}else{
				Pair pair = (Pair)unit;
				assert (params.getUseSelfRelation() && pair.getW1().getId() <= pair.getW2().getId()) 
				|| (!params.getUseSelfRelation() && pair.getW1().getId() < pair.getW2().getId()): pair.getW1().getId() +":"+ pair.getW2().getId();
				wordPairs[pair.getW1().getId()][pair.getW2().getId()] = index;
			}
		}
	}

	public JointInstance(JointInstance instance){
		super(instance.params, instance.sequence, instance.goldLabel);
		int nwords = instance.words.length;
		words = new int[nwords];
		wordPairs = new int[nwords][nwords];
		System.arraycopy(instance.words, 0, words, 0, nwords);
		for(int i = 0;i < nwords;i++){
			System.arraycopy(instance.wordPairs[i], 0, wordPairs[i], 0, nwords);
		}
		if(instance.goldScores != null){
			int length = instance.goldScores.length;
			goldScores = new double[length];
			System.arraycopy(instance.goldScores, 0, goldScores, 0, length);
		}
	}
	
	@Override
	public void sort() {
		//TODO do something??
		assert false;
		return;
	}
	public int getNumWords(){
		return words.length;
	}
	
	public int getWord(int w){
		//word index in sentence => word index in sequence
		return words[w];
	}
	
	public int getWordPair(int w1, int w2){
		assert w1 <= w2;
		return wordPairs[w1][w2];
	}
	

	
	public void sort(Model model, Label y, int currentIndex, boolean test) {	
		if(params.useEasyFirst()){
			sortEasyFirst(model, y, currentIndex, test);	
		}else if(params.useDifficultFirst()){
			sortDifficultFirst(model, y, currentIndex, test);	
		}else if(params.useRandomFirst()){
			sortRandomFirst(model, y, currentIndex, test);	
		}else if(params.useEasyEntityFirst()){
			sortEasyEntityFirst(model, y, currentIndex, test);	
		}else if(params.useEasyCloseFirst()){
			sortEasyCloseFirst(model, y, currentIndex, test);	
		}
	}
	
	private class EntFirstMeasure implements Comparable<EntFirstMeasure> {
		private int type;
		private double score;
		private int distance;
		private int index;		
		
		public EntFirstMeasure(int type, double score, int distance, int index) {
			this.type = type;
			this.score = score;
			this.distance = distance;
			this.index = index;
		}

		@Override
		public int compareTo(EntFirstMeasure dist) {
			if(type != dist.type){
				return type < dist.type ? -1 : 1;//smaller better
			}
			if(score != dist.score){
				return score < dist.score ? 1 : -1;//larger better
			}

			if(distance != dist.distance){
				return distance < dist.distance ? -1 : 1;//smaller better
			}
			return index - dist.index; // smaller better
		}

		@Override
		public String toString() {
			return "EntFirstMeasure [score=" + score + ", index=" + index + "]";
		}
	}
	
	private void sortEasyCloseFirst(Model model, Label y, int currentIndex, boolean test) {	
		Multimap<EntFirstMeasure, Integer> values = TreeMultimap.create(Ordering.natural(), Ordering.natural());
		for(int target = currentIndex;target < size();target++){
			if(sequence.get(target) instanceof Word){
				Word word = (Word)sequence.get(target);
				List<Double> scores = new Vector<Double>();
				Collection<LabelUnit> labels = word.getPossibleLabels(y, this);
				if(labels.size() == 1){
					values.put(new EntFirstMeasure(0, Double.POSITIVE_INFINITY, 0, word.getId()), target);					
				}else{
					for(LabelUnit label:labels){
						double score = model.evaluate(model.getFeatureGenerator().calculateFeature(this, y, currentIndex - 1, target, label), test);
						scores.add(score);
					}
					int size = scores.size();
					assert word.getId() < words.length;
					Collections.sort(scores);
					values.put(new EntFirstMeasure(0, scores.get(size - 1), 0, word.getId()), target);					
				}
			}else{
				Pair pair = (Pair)sequence.get(target);
				List<Double> scores = new Vector<Double>();
				Collection<LabelUnit> labels = pair.getPossibleLabels(y, this);
				int pairIdx = (pair.getW1().getId() + 1) * words.length + pair.getW2().getId() + 1;
				if(labels.size() == 1){
					values.put(new EntFirstMeasure(1+Math.abs(pair.getW1().getId()-pair.getW2().getId()), Double.POSITIVE_INFINITY, Math.abs(pair.getW1().getId()-pair.getW2().getId()), pairIdx), target);					
				}else{
					for(LabelUnit label:labels){
						double score = model.evaluate(model.getFeatureGenerator().calculateFeature(this, y, currentIndex - 1, target, label), test);
						scores.add(score);
					}
					int size = scores.size();
					Collections.sort(scores);
					values.put(new EntFirstMeasure(1+Math.abs(pair.getW1().getId()-pair.getW2().getId()), scores.get(size - 1), Math.abs(pair.getW1().getId()-pair.getW2().getId()), pairIdx), target);
				}
			}
		}

		Label goldLabel = new Label(params);
		Sequence sequence = new JointSequence(params);
		int newIndex = 0;
		for(int index = 0; index < currentIndex; index++){
			SequenceUnit unit = this.sequence.get(index);
			goldLabel.add(this.goldLabel.getLabel(index));
			sequence.add(unit);
			newIndex++;
		}	
		for(int index:values.values()){
			goldLabel.add(this.goldLabel.getLabel(index));
			sequence.add(this.sequence.get(index));
			SequenceUnit unit = this.sequence.get(index);
			if(unit instanceof Word){
				words[((Word)unit).getId()] = newIndex;
			}else{
				Pair pair = (Pair)unit;
				assert (params.getUseSelfRelation() && pair.getW1().getId() <= pair.getW2().getId()) 
				|| (!params.getUseSelfRelation() && pair.getW1().getId() < pair.getW2().getId());
				wordPairs[pair.getW1().getId()][pair.getW2().getId()] = newIndex;
			}
			newIndex++;
		}
		this.goldLabel = goldLabel;
		this.sequence = sequence;
	}
	
	
	
	private void sortEasyEntityFirst(Model model, Label y, int currentIndex, boolean test) {	
		Multimap<EntFirstMeasure, Integer> values = TreeMultimap.create(Ordering.natural(), Ordering.natural());
		for(int target = currentIndex;target < size();target++){
			if(sequence.get(target) instanceof Word){
				Word word = (Word)sequence.get(target);
				List<Double> scores = new Vector<Double>();
				Collection<LabelUnit> labels = word.getPossibleLabels(y, this);
				if(labels.size() == 1){
					values.put(new EntFirstMeasure(0, Double.POSITIVE_INFINITY, 0, word.getId()), target);					
				}else{
					for(LabelUnit label:labels){
						double score = model.evaluate(model.getFeatureGenerator().calculateFeature(this, y, currentIndex - 1, target, label), test);
						scores.add(score);
					}
					int size = scores.size();
					assert word.getId() < words.length;
					Collections.sort(scores);
					values.put(new EntFirstMeasure(0, scores.get(size - 1), 0, word.getId()), target);					
				}
			}else{
				Pair pair = (Pair)sequence.get(target);
				List<Double> scores = new Vector<Double>();
				Collection<LabelUnit> labels = pair.getPossibleLabels(y, this);
				int pairIdx = (pair.getW1().getId() + 1) * words.length + pair.getW2().getId() + 1;
				if(labels.size() == 1){
					values.put(new EntFirstMeasure(1, Double.POSITIVE_INFINITY, Math.abs(pair.getW1().getId()-pair.getW2().getId()), pairIdx), target);					
				}else{
					for(LabelUnit label:labels){
						double score = model.evaluate(model.getFeatureGenerator().calculateFeature(this, y, currentIndex - 1, target, label), test);
						scores.add(score);
					}
					int size = scores.size();
					Collections.sort(scores);
					values.put(new EntFirstMeasure(1, scores.get(size - 1), Math.abs(pair.getW1().getId()-pair.getW2().getId()), pairIdx), target);
				}
			}
		}

		Label goldLabel = new Label(params);
		Sequence sequence = new JointSequence(params);
		int newIndex = 0;
		for(int index = 0; index < currentIndex; index++){
			SequenceUnit unit = this.sequence.get(index);
			goldLabel.add(this.goldLabel.getLabel(index));
			sequence.add(unit);
			newIndex++;
		}	
		for(int index:values.values()){
			goldLabel.add(this.goldLabel.getLabel(index));
			sequence.add(this.sequence.get(index));
			SequenceUnit unit = this.sequence.get(index);
			if(unit instanceof Word){
				words[((Word)unit).getId()] = newIndex;
			}else{
				Pair pair = (Pair)unit;
				assert (params.getUseSelfRelation() && pair.getW1().getId() <= pair.getW2().getId()) 
				|| (!params.getUseSelfRelation() && pair.getW1().getId() < pair.getW2().getId());
				wordPairs[pair.getW1().getId()][pair.getW2().getId()] = newIndex;
			}
			newIndex++;
		}
		this.goldLabel = goldLabel;
		this.sequence = sequence;
	}
	
	
	private void sortRandomFirst(Model model, Label y, int currentIndex, boolean test) {	
		List<Integer> indicies = new Vector<Integer>();
		for(int target = currentIndex;target < size();target++){
			indicies.add(target);
		}
		Collections.shuffle(indicies);
		Label goldLabel = new Label(params);
		Sequence sequence = new JointSequence(params);
		int newIndex = 0;
		for(int index = 0; index < currentIndex; index++){
			SequenceUnit unit = this.sequence.get(index);
			goldLabel.add(this.goldLabel.getLabel(index));
			sequence.add(unit);
			newIndex++;
		}	
		for(int index:indicies){
			goldLabel.add(this.goldLabel.getLabel(index));
			sequence.add(this.sequence.get(index));
			SequenceUnit unit = this.sequence.get(index);
			if(unit instanceof Word){
				words[((Word)unit).getId()] = newIndex;
			}else{
				Pair pair = (Pair)unit;
				assert (params.getUseSelfRelation() && pair.getW1().getId() <= pair.getW2().getId()) 
				|| (!params.getUseSelfRelation() && pair.getW1().getId() < pair.getW2().getId());
				wordPairs[pair.getW1().getId()][pair.getW2().getId()] = newIndex;
			}
			newIndex++;
		}
		this.goldLabel = goldLabel;
		this.sequence = sequence;
	}
	
	private class Measure implements Comparable<Measure> {
		private double score;
		private int distance;
		private int index;		
		
		public Measure(double score, int distance, int index) {
			this.score = score;
			this.distance = distance;
			this.index = index;
		}

		@Override
		public int compareTo(Measure dist) {
			if(score != dist.score){
				return score < dist.score ? 1 : -1;//larger better
			}

			if(distance != dist.distance){
				return distance < dist.distance ? -1 : 1;//smaller better
			}
			return index - dist.index; // smaller better
		}

		@Override
		public String toString() {
			return "Distance [score=" + score + ", index=" + index + "]";
		}
	}
	
	private void sortDifficultFirst(Model model, Label y, int currentIndex, boolean test) {	
		Multimap<Measure, Integer> values = TreeMultimap.create(Ordering.natural().reverse(), Ordering.natural());
		for(int target = currentIndex;target < size();target++){
			if(sequence.get(target) instanceof Word){
				Word word = (Word)sequence.get(target);
				List<Double> scores = new Vector<Double>();
				Collection<LabelUnit> labels = word.getPossibleLabels(y, this);
				if(labels.size() == 1){
					values.put(new Measure(Double.POSITIVE_INFINITY, 0, word.getId()), target);					
				}else{
					for(LabelUnit label:labels){
						double score = model.evaluate(model.getFeatureGenerator().calculateFeature(this, y, currentIndex - 1, target, label), test);
						scores.add(score);
					}
					int size = scores.size();
					assert word.getId() < words.length;
					Collections.sort(scores);
					values.put(new Measure(scores.get(size - 1), 0, word.getId()), target);					
				}
			}else{
				Pair pair = (Pair)sequence.get(target);
				List<Double> scores = new Vector<Double>();
				Collection<LabelUnit> labels = pair.getPossibleLabels(y, this);
				int pairIdx = (pair.getW1().getId() + 1) * words.length + pair.getW2().getId() + 1;
				if(labels.size() == 1){
					values.put(new Measure(Double.POSITIVE_INFINITY, Math.abs(pair.getW1().getId()-pair.getW2().getId()), pairIdx), target);					
				}else{
					for(LabelUnit label:labels){
						double score = model.evaluate(model.getFeatureGenerator().calculateFeature(this, y, currentIndex - 1, target, label), test);
						scores.add(score);
					}
					int size = scores.size();
					Collections.sort(scores);
					values.put(new Measure(scores.get(size - 1), Math.abs(pair.getW1().getId()-pair.getW2().getId()), pairIdx), target);
				}
			}
		}

		Label goldLabel = new Label(params);
		Sequence sequence = new JointSequence(params);
		int newIndex = 0;
		for(int index = 0; index < currentIndex; index++){
			SequenceUnit unit = this.sequence.get(index);
			goldLabel.add(this.goldLabel.getLabel(index));
			sequence.add(unit);
			newIndex++;
		}	
		for(int index:values.values()){
			goldLabel.add(this.goldLabel.getLabel(index));
			sequence.add(this.sequence.get(index));
			SequenceUnit unit = this.sequence.get(index);
			if(unit instanceof Word){
				words[((Word)unit).getId()] = newIndex;
			}else{
				Pair pair = (Pair)unit;
				assert (params.getUseSelfRelation() && pair.getW1().getId() <= pair.getW2().getId()) 
				|| (!params.getUseSelfRelation() && pair.getW1().getId() < pair.getW2().getId());
				wordPairs[pair.getW1().getId()][pair.getW2().getId()] = newIndex;
			}
			newIndex++;
		}
		this.goldLabel = goldLabel;
		this.sequence = sequence;
	}
	
	private void sortEasyFirst(Model model, Label y, int currentIndex, boolean test) {	
		assert params.useEasyFirst();
		int next = -1;
		for(int target = currentIndex;target < size();target++){
			if(sequence.get(target) instanceof Word){
				Word word = (Word)sequence.get(target);
				Collection<LabelUnit> labels = word.getPossibleLabels(y, this);
				if(labels.size() == 1){
					next = target;
					break;
				}							
			}else{
				Pair pair = (Pair)sequence.get(target);
				Collection<LabelUnit> labels = pair.getPossibleLabels(y, this);
				if(labels.size() == 1){
					next = target;
					break;
				}
			}		
		}
		if(next < 0){
			// no immediate assignment
			Multimap<Measure, Integer> values = TreeMultimap.create(Ordering.natural(), Ordering.natural());
			for(int target = currentIndex;target < size();target++){
				if(sequence.get(target) instanceof Word){
					Word word = (Word)sequence.get(target);
					List<Double> scores = new Vector<Double>();
					Collection<LabelUnit> labels = word.getPossibleLabels(y, this);
					for(LabelUnit label:labels){
						double score = model.evaluate(model.getFeatureGenerator().calculateFeature(this, y, currentIndex - 1, target, label), test);
						scores.add(score);
					}
					int size = scores.size();
					assert word.getId() < words.length;
					Collections.sort(scores);
					values.put(new Measure(scores.get(size - 1), 0, word.getId()), target);					
				}else{
					Pair pair = (Pair)sequence.get(target);
					List<Double> scores = new Vector<Double>();
					Collection<LabelUnit> labels = pair.getPossibleLabels(y, this);
					int pairIdx = (pair.getW1().getId() + 1) * words.length + pair.getW2().getId() + 1;
					for(LabelUnit label:labels){
						double score = model.evaluate(model.getFeatureGenerator().calculateFeature(this, y, currentIndex - 1, target, label), test);
						scores.add(score);
					}
					int size = scores.size();
					Collections.sort(scores);
					values.put(new Measure(scores.get(size - 1), Math.abs(pair.getW1().getId()-pair.getW2().getId()), pairIdx), target);
				}
			}

			Label goldLabel = new Label(params);
			Sequence sequence = new JointSequence(params);
			int newIndex = 0;
			for(int index = 0; index < currentIndex; index++){
				SequenceUnit unit = this.sequence.get(index);
				goldLabel.add(this.goldLabel.getLabel(index));
				sequence.add(unit);
				newIndex++;
			}	
			for(int index:values.values()){
				goldLabel.add(this.goldLabel.getLabel(index));
				sequence.add(this.sequence.get(index));
				SequenceUnit unit = this.sequence.get(index);
				if(unit instanceof Word){
					words[((Word)unit).getId()] = newIndex;
				}else{
					Pair pair = (Pair)unit;
					assert (params.getUseSelfRelation() && pair.getW1().getId() <= pair.getW2().getId()) 
					|| (!params.getUseSelfRelation() && pair.getW1().getId() < pair.getW2().getId());
					wordPairs[pair.getW1().getId()][pair.getW2().getId()] = newIndex;
				}
				newIndex++;
			}
			this.goldLabel = goldLabel;
			this.sequence = sequence;
		}else{
			Label goldLabel = new Label(params);
			Sequence sequence = new JointSequence(params);
			int newIndex = 0;
			for(int index = 0; index < currentIndex; index++){
				SequenceUnit unit = this.sequence.get(index);
				goldLabel.add(this.goldLabel.getLabel(index));
				sequence.add(unit);
				newIndex++;
			}
			{
				goldLabel.add(this.goldLabel.getLabel(next));
				sequence.add(this.sequence.get(next));
				SequenceUnit unit = this.sequence.get(next);
				if(unit instanceof Word){
					words[((Word)unit).getId()] = newIndex;
				}else{
					Pair pair = (Pair)unit;
					assert (params.getUseSelfRelation() && pair.getW1().getId() <= pair.getW2().getId()) 
					|| (!params.getUseSelfRelation() && pair.getW1().getId() < pair.getW2().getId());
					wordPairs[pair.getW1().getId()][pair.getW2().getId()] = newIndex;
				}
				newIndex++;
			}		
			for(int index = currentIndex;index < size();index++){
				if(index == next){
					continue;
				}
				goldLabel.add(this.goldLabel.getLabel(index));
				sequence.add(this.sequence.get(index));
				SequenceUnit unit = this.sequence.get(index);
				if(unit instanceof Word){
					words[((Word)unit).getId()] = newIndex;
				}else{
					Pair pair = (Pair)unit;
					assert (params.getUseSelfRelation() && pair.getW1().getId() <= pair.getW2().getId()) 
					|| (!params.getUseSelfRelation() && pair.getW1().getId() < pair.getW2().getId());
					wordPairs[pair.getW1().getId()][pair.getW2().getId()] = newIndex;
				}
				newIndex++;
			}
			this.goldLabel = goldLabel;
			this.sequence = sequence;
		}
	}
	
	@Override
	public void sort(Model model, boolean test) {	
		if(params.useEasyFirst()){
			sortEasyFirst(model, test, false);	
		}else if(params.useDifficultFirst()){
			sortEasyFirst(model,test, true);	
		}else if(params.useRandomFirst()){
			sortRandomFirst(model, test);	
		}else if(params.useEasyEntityFirst()){
			sortEasyEntityFirst(model, test);	
		}else if(params.useEasyCloseFirst()){
			sortEasyCloseFirst(model, test);	
		}
	}
	
	public void sortEasyCloseFirst(Model model, boolean test) {
		Multimap<EntFirstMeasure, Integer> values = TreeMultimap.create(Ordering.natural(), Ordering.natural());
		for(int index = 0;index < size();index++){
			if(sequence.get(index) instanceof Word){
				Word word = (Word)sequence.get(index);
				List<Double> scores = new Vector<Double>();
				for(LabelUnit label:word.getPossibleLabels(new Label(params), this)){
					double score = model.evaluate(model.getFeatureGenerator().calculateFeature(this, null, index, label, true), test);
					scores.add(score);
				}
				int size = scores.size();
				assert word.getId() < words.length;
				if(size > 1){
					Collections.sort(scores);
					values.put(new EntFirstMeasure(0, scores.get(size - 1), -1, word.getId()), index);
				}else{
					values.put(new EntFirstMeasure(0, Double.POSITIVE_INFINITY, -1, word.getId()), index);
				}							
			}else{
				Pair pair = (Pair)sequence.get(index);
				List<Double> scores = new Vector<Double>();
				for(LabelUnit label:pair.getPossibleLabels(new Label(params), this)){
					double score = model.evaluate(model.getFeatureGenerator().calculateFeature(this, null, index, label, true), test);
					scores.add(score);
				}
				int size = scores.size();
				int pairIdx = (pair.getW1().getId() + 1) * words.length + pair.getW2().getId() + 1;
				if(size > 1){
					Collections.sort(scores);
					values.put(new EntFirstMeasure(1+Math.abs(pair.getW1().getId() - pair.getW2().getId()), scores.get(size - 1), Math.abs(pair.getW1().getId() - pair.getW2().getId()), pairIdx), index);
				}else{
					values.put(new EntFirstMeasure(1+Math.abs(pair.getW1().getId() - pair.getW2().getId()), Double.POSITIVE_INFINITY, 0, pairIdx), index);
				}
			}
		}
		Label goldLabel = new Label(params);
		Sequence sequence = new JointSequence(params);
		assert size() == values.values().size();
		int newIndex = 0;
		for(int index:values.values()){
			goldLabel.add(this.goldLabel.getLabel(index));
			sequence.add(this.sequence.get(index));
			SequenceUnit unit = this.sequence.get(index);
			if(unit instanceof Word){
				words[((Word)unit).getId()] = newIndex;
			}else{
				Pair pair = (Pair)unit;
				assert (params.getUseSelfRelation() && pair.getW1().getId() <= pair.getW2().getId()) 
				|| (!params.getUseSelfRelation() && pair.getW1().getId() < pair.getW2().getId());
				wordPairs[pair.getW1().getId()][pair.getW2().getId()] = newIndex;
			}
			newIndex++;
		}
		this.goldLabel = goldLabel;
		this.sequence = sequence;
	}
	
	
	public void sortEasyEntityFirst(Model model, boolean test) {
		Multimap<EntFirstMeasure, Integer> values = TreeMultimap.create(Ordering.natural(), Ordering.natural());
		for(int index = 0;index < size();index++){
			if(sequence.get(index) instanceof Word){
				Word word = (Word)sequence.get(index);
				List<Double> scores = new Vector<Double>();
				for(LabelUnit label:word.getPossibleLabels(new Label(params), this)){
					double score = model.evaluate(model.getFeatureGenerator().calculateFeature(this, null, index, label, true), test);
					scores.add(score);
				}
				int size = scores.size();
				assert word.getId() < words.length;
				if(size > 1){
					Collections.sort(scores);
					values.put(new EntFirstMeasure(0, scores.get(size - 1), -1, word.getId()), index);
				}else{
					values.put(new EntFirstMeasure(0, Double.POSITIVE_INFINITY, -1, word.getId()), index);
				}							
			}else{
				Pair pair = (Pair)sequence.get(index);
				List<Double> scores = new Vector<Double>();
				for(LabelUnit label:pair.getPossibleLabels(new Label(params), this)){
					double score = model.evaluate(model.getFeatureGenerator().calculateFeature(this, null, index, label, true), test);
					scores.add(score);
				}
				int size = scores.size();
				int pairIdx = (pair.getW1().getId() + 1) * words.length + pair.getW2().getId() + 1;
				if(size > 1){
					Collections.sort(scores);
					values.put(new EntFirstMeasure(1, scores.get(size - 1), Math.abs(pair.getW1().getId() - pair.getW2().getId()), pairIdx), index);
				}else{
					values.put(new EntFirstMeasure(1, Double.POSITIVE_INFINITY, 0, pairIdx), index);
				}
			}
		}
		Label goldLabel = new Label(params);
		Sequence sequence = new JointSequence(params);
		assert size() == values.values().size();
		int newIndex = 0;
		for(int index:values.values()){
			goldLabel.add(this.goldLabel.getLabel(index));
			sequence.add(this.sequence.get(index));
			SequenceUnit unit = this.sequence.get(index);
			if(unit instanceof Word){
				words[((Word)unit).getId()] = newIndex;
			}else{
				Pair pair = (Pair)unit;
				assert (params.getUseSelfRelation() && pair.getW1().getId() <= pair.getW2().getId()) 
				|| (!params.getUseSelfRelation() && pair.getW1().getId() < pair.getW2().getId());
				wordPairs[pair.getW1().getId()][pair.getW2().getId()] = newIndex;
			}
			newIndex++;
		}
		this.goldLabel = goldLabel;
		this.sequence = sequence;
	}
	
	
	public void sortRandomFirst(Model model, boolean test) {	
		List<Integer> indicies = new Vector<Integer>();
		for(int target = 0;target < size();target++){
			indicies.add(target);
		}
		Collections.shuffle(indicies);		
		Label goldLabel = new Label(params);
		Sequence sequence = new JointSequence(params);
		int newIndex = 0;
		for(int index:indicies){
			goldLabel.add(this.goldLabel.getLabel(index));
			sequence.add(this.sequence.get(index));
			SequenceUnit unit = this.sequence.get(index);
			if(unit instanceof Word){
				words[((Word)unit).getId()] = newIndex;
			}else{
				Pair pair = (Pair)unit;
				assert (params.getUseSelfRelation() && pair.getW1().getId() <= pair.getW2().getId()) 
				|| (!params.getUseSelfRelation() && pair.getW1().getId() < pair.getW2().getId());
				wordPairs[pair.getW1().getId()][pair.getW2().getId()] = newIndex;
			}
			newIndex++;
		}
		this.goldLabel = goldLabel;
		this.sequence = sequence;
	}
	
	public void sortEasyFirst(Model model, boolean test, boolean reverse) {	
		assert params.useEasyFirst() || params.useDifficultFirst();
		Multimap<Measure, Integer> values;
		if(reverse){
			values = TreeMultimap.create(Ordering.natural().reverse(), Ordering.natural());
		}else{
			values = TreeMultimap.create(Ordering.natural(), Ordering.natural());
		}
		for(int index = 0;index < size();index++){
			if(sequence.get(index) instanceof Word){
				Word word = (Word)sequence.get(index);
				List<Double> scores = new Vector<Double>();
				for(LabelUnit label:word.getPossibleLabels(new Label(params), this)){
					double score = model.evaluate(model.getFeatureGenerator().calculateFeature(this, null, index, label, true), test);
					scores.add(score);
				}
				int size = scores.size();
				assert word.getId() < words.length;
				if(size > 1){
					Collections.sort(scores);
					values.put(new Measure(scores.get(size - 1), -1, word.getId()), index);
				}else{
					values.put(new Measure(Double.POSITIVE_INFINITY, -1, word.getId()), index);
				}							
			}else{
				Pair pair = (Pair)sequence.get(index);
				List<Double> scores = new Vector<Double>();
				for(LabelUnit label:pair.getPossibleLabels(new Label(params), this)){
					double score = model.evaluate(model.getFeatureGenerator().calculateFeature(this, null, index, label, true), test);
					scores.add(score);
				}
				int size = scores.size();
				int pairIdx = (pair.getW1().getId() + 1) * words.length + pair.getW2().getId() + 1;
				if(size > 1){
					Collections.sort(scores);
					values.put(new Measure(scores.get(size - 1), Math.abs(pair.getW1().getId() - pair.getW2().getId()), pairIdx), index);
				}else{
					values.put(new Measure(Double.POSITIVE_INFINITY, 0, pairIdx), index);
				}
			}
		}
		Label goldLabel = new Label(params);
		Sequence sequence = new JointSequence(params);
		assert size() == values.values().size();
		int newIndex = 0;
		for(int index:values.values()){
			goldLabel.add(this.goldLabel.getLabel(index));
			sequence.add(this.sequence.get(index));
			SequenceUnit unit = this.sequence.get(index);
			if(unit instanceof Word){
				words[((Word)unit).getId()] = newIndex;
			}else{
				Pair pair = (Pair)unit;
				assert (params.getUseSelfRelation() && pair.getW1().getId() <= pair.getW2().getId()) 
				|| (!params.getUseSelfRelation() && pair.getW1().getId() < pair.getW2().getId());
				wordPairs[pair.getW1().getId()][pair.getW2().getId()] = newIndex;
			}
			newIndex++;
		}
		this.goldLabel = goldLabel;
		this.sequence = sequence;
	}

	@Override
	public void cacheFeatures() {
		for(int i = 0;i < size();i++){
			if(sequence.get(i) instanceof Pair){
				Pair pair = (Pair)sequence.get(i);
				Node w1 = pair.getW1().getWord();
				if(w1.getCache() == null){
					w1.setCache(new JointFeatureCache(params, w1));
					w1.getCache().calcNodeFeatures();
				}
				Node w2 = pair.getW2().getWord();
				if(w2.getCache() == null){
					w2.setCache(new JointFeatureCache(params, w2));
					w2.getCache().calcNodeFeatures();
				}
				w1.getCache().calcPathFeatures(w2);
				//w2.getCache().calcPathFeatures(w1);
			}
			if(sequence.get(i) instanceof Word){
				Node w = ((Word)sequence.get(i)).getWord();
				if(w.getCache() == null){
					w.setCache(new JointFeatureCache(params, w));
					w.getCache().calcNodeFeatures();
				}				
			}
		}
	}

	public Collection<LabelUnit> getCandidateLabels(Label label, int index) {
		assert index < goldLabel.size();
		SequenceUnit unit = sequence.get(index);
		if(unit instanceof Word){
			Word word = (Word)unit;
			return word.getPossibleLabels(label, this);
		}else{
			assert unit instanceof Pair;
			Pair pair = (Pair)unit;
			return pair.getPossibleLabels(label, this);
		}
	}
	
	
	public String getGoldType(Label label, int id){
		String goldPosition = ((WordLabelUnit)getGoldLabel().getLabel(getWord(id))).getPosition();
		int size = label.size();
		if(goldPosition.equals("B")){
			for(int i = id+1;;++i){
				String pos = ((WordLabelUnit)getGoldLabel().getLabel(getWord(i))).getPosition();					
				int wIdx = getWord(i);
				if(wIdx < size){
					return ((WordLabelUnit)label.getLabel(wIdx)).getType();
				}
				if(pos.equals("L")){
					break;
				}
			}
		}else if(goldPosition.equals("L")){
			for(int i = id-1;;--i){
				String pos = ((WordLabelUnit)getGoldLabel().getLabel(getWord(i))).getPosition();					
				int wIdx = getWord(i);
				if(wIdx < size){
					return ((WordLabelUnit)label.getLabel(wIdx)).getType();
				}
				if(pos.equals("B")){
					break;
				}
			}				
		}else if(goldPosition.equals("I")){
			for(int i = id-1;;--i){
				String pos = ((WordLabelUnit)getGoldLabel().getLabel(getWord(i))).getPosition();						
				int wIdx = getWord(i);
				if(wIdx < size){
					return ((WordLabelUnit)label.getLabel(wIdx)).getType();
				}
				if(pos.equals("B")){
					break;
				}
			}
			for(int i = id+1;;++i){
				String pos = ((WordLabelUnit)getGoldLabel().getLabel(getWord(i))).getPosition();
				int wIdx = getWord(i);
				if(wIdx < size){
					return ((WordLabelUnit)label.getLabel(wIdx)).getType();
				}
				if(pos.equals("L")){
					break;
				}
			}
		}
		return "";
	}

	@Override
	public void clearCachedFeatures() {
		for(int i = 0;i < size();i++){
			if(sequence.get(i) instanceof Pair){
				Pair pair = (Pair)sequence.get(i);
				Node w1 = pair.getW1().getWord();
				if(w1.getCache() != null){
					w1.setCache(null);
				}
				Node w2 = pair.getW2().getWord();
				if(w2.getCache() != null){
					w2.setCache(null);
				}
			}
		}
	}

}
