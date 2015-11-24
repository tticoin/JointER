package data.nlp.joint;

import java.util.Arrays;
import java.util.List;
import java.util.Vector;

import model.FeatureGenerator;
import model.SparseFeatureVector;
import model.StringSparseVector;
import config.Parameters;
import data.Instance;
import data.Label;
import data.LabelUnit;
import data.nlp.Node;

public class JointFeatureGenerator extends FeatureGenerator {
	
	public JointFeatureGenerator(Parameters params) {
		super(params);
	}
	
	@Override
	public void init() {}
	
	@Override
	public SparseFeatureVector calculateFeature(Instance instance, Label y, int lastIndex, int target, LabelUnit candidateLabelUnit, boolean local) {
		//TODO cache strings
		SparseFeatureVector fv = new SparseFeatureVector(params);
		if(instance.getSequence().get(target) instanceof Word){
			Word word = (Word)instance.getSequence().get(target);
			String candidateLabel = ((WordLabelUnit)candidateLabelUnit).getLabel();
			assert word.getWord().getCache() != null: word.getWord().getId()+":"+word.getWord().getRealBase();
			fv.add(word.getWord().getCache().getNodeFeatures(), candidateLabel);
			if(params.getUseGlobalFeatures() && !local){
				SparseFeatureVector globalFv = calcWordGlobalFeatures(word, instance, y, lastIndex, (WordLabelUnit)candidateLabelUnit);
				assert params.getAverageNumWords() > 0.;
				globalFv.scale(params.getGlobalWeight());
				fv.add(globalFv, "GLOBAL");
			}
		}else{
			assert instance.getSequence().get(target) instanceof Pair;
			Pair pair = (Pair)instance.getSequence().get(target);
			PairLabelUnit candidatePairLabelUnit = (PairLabelUnit)candidateLabelUnit;
			Node w1 = pair.getW1().getWord();
			Node w2 = pair.getW2().getWord();
			fv.add(w1.getCache().getFullPathFeatures(w2), candidatePairLabelUnit.getLabel().concat("PATH"));
			if(params.getUseGlobalFeatures() && !local){
				SparseFeatureVector globalFv = calcPairGlobalFeatures(pair, instance, y, lastIndex, candidatePairLabelUnit);
				assert params.getAverageNumWords() > 0.;
				globalFv.scale(params.getGlobalWeight());
				fv.add(globalFv, "GLOBAL");
			}
		}			
		assert fv.size() > 0;
		return fv;
	}
	

	protected void addBigramToFV(StringSparseVector fv, String label1, String base1, String pos1, String label2, String base2, String pos2){
		StringSparseVector bigramFv = new StringSparseVector(params);
		bigramFv.add(label1.concat(base1).concat(label2).concat(base2), 1.);
		bigramFv.add(label1.concat(base1).concat(label2), 1.);
		bigramFv.add(label1.concat(label2).concat(base2), 1.);
		bigramFv.add(label1.concat(pos1).concat(label2).concat(pos2), 1.);
		bigramFv.add(label1.concat(pos1).concat(label2), 1.);
		bigramFv.add(label1.concat(label2).concat(pos2), 1.);
		bigramFv.add(label1.concat(label2), 1.);
		bigramFv.normalize(1.);
		fv.add(bigramFv);
	}
	
	protected void addTrigramToFV(StringSparseVector fv, String label1, String base1, String pos1, String label2, String base2, String pos2, String label3, String base3, String pos3){
		StringSparseVector trigramFv = new StringSparseVector(params);
		trigramFv.add(label1.concat(label2).concat(label3), 1.);
		trigramFv.add(label1.concat(base1).concat(label2).concat(label3), 1.);
		trigramFv.add(label1.concat(label2).concat(base2).concat(label3), 1.);
		trigramFv.add(label1.concat(label2).concat(label3).concat(base3), 1.);
		trigramFv.add(label1.concat(pos1).concat(label2).concat(label3), 1.);
		trigramFv.add(label1.concat(label2).concat(pos2).concat(label3), 1.);
		trigramFv.add(label1.concat(label2).concat(label3).concat(pos3), 1.);
		trigramFv.normalize(1.);
		fv.add(trigramFv);
	}
	
	protected List<Word> getEntityWords(Word word, Instance instance, Label y, int lastIndex, WordLabelUnit candidateLabelUnit){
		List<Word> words = new Vector<Word>();
		if(candidateLabelUnit.isNegative()){
			return words;
		}
		int targetId = word.getId();
		if(candidateLabelUnit.getPosition().equals("U")){
			words.add(word);
		}else if(candidateLabelUnit.getPosition().equals("L")){
			words.add(word);
			boolean found = false;
			for(int i = targetId - 1;i >= 0;--i){
				int seqIndex = ((JointInstance)instance).getWord(i);	
				if(seqIndex >= lastIndex){
					break;
				}
				String pos = ((WordLabelUnit)y.getLabel(seqIndex)).getPosition();
				if(pos.equals("B")){
					words.add(0, word);
					found = true;
				}else if(pos.equals("I")){
					words.add(0, word);
				}
			}
			if(!found){
				words.clear();
			}			
		}else if(candidateLabelUnit.getPosition().equals("I")){
			words.add(word);
			boolean leftFound = false;
			for(int i = targetId - 1;i >= 0;--i){
				int seqIndex = ((JointInstance)instance).getWord(i);	
				if(seqIndex >= lastIndex){
					break;
				}
				String pos = ((WordLabelUnit)y.getLabel(seqIndex)).getPosition();
				if(pos.equals("B")){
					words.add(0, word);
					leftFound = true;
				}else if(pos.equals("I")){
					words.add(0, word);
				}
			}
			if(!leftFound){
				words.clear();
			}else{
				boolean rightFound = false;
				int nwords = ((JointInstance)instance).getNumWords();
				for(int i = targetId + 1;i < nwords;++i){
					int seqIndex = ((JointInstance)instance).getWord(i);	
					if(seqIndex >= lastIndex){
						break;
					}
					String pos = ((WordLabelUnit)y.getLabel(seqIndex)).getPosition();
					if(pos.equals("L")){
						words.add(word);
						rightFound = true;
					}else if(pos.equals("I")){
						words.add(word);
					}
				}	
				if(!rightFound){
					words.clear();
				}
			}
		}else if(candidateLabelUnit.getPosition().equals("B")){
			boolean found = false;
			int nwords = ((JointInstance)instance).getNumWords();
			for(int i = targetId + 1;i < nwords;++i){
				int seqIndex = ((JointInstance)instance).getWord(i);	
				if(seqIndex >= lastIndex){
					break;
				}
				String pos = ((WordLabelUnit)y.getLabel(seqIndex)).getPosition();
				if(pos.equals("L")){
					words.add(word);
					found = true;
				}else if(pos.equals("I")){
					words.add(word);
				}
			}	
			if(!found){
				words.clear();
			}
		}
		return words;
	}
	
	protected String entityString(List<Word> entityWords){
		StringBuffer sb = new StringBuffer();
		for(Word entityWord:entityWords){
			sb.append(entityWord.getWord().getRealBase());
			sb.append(" ");
		}
		return sb.toString().trim();
	}
		
	protected SparseFeatureVector calcWordGlobalFeatures(Word word, Instance instance, Label y, int lastIndex, WordLabelUnit candidateLabelUnit){
		SparseFeatureVector globalFv = new SparseFeatureVector(params);
		StringSparseVector wordPairFv = new StringSparseVector(params);
		StringSparseVector wordFv = new StringSparseVector(params);
		
		String candidateLabel = candidateLabelUnit.getLabel();
		String wBase = word.getWord().getRealBase();
		String wPOS = word.getWord().getPOS();
		
		int[] contextWords = new int[4];
		Arrays.fill(contextWords, -1);
		
		for(int i = 0;i <= lastIndex;i++){
			if(y.getLabel(i) instanceof PairLabelUnit){
				if(y.getLabel(i).isNegative())continue;
				if(candidateLabelUnit.isNegative()){
					continue;
				}
				Pair pair = (Pair)instance.getSequence().get(i);
				if(pair.getW1().getId() == word.getId()){
					String pairLabel = ((PairLabelUnit)y.getLabel(i)).getLabel();					
					wordPairFv.add(candidateLabel.concat(pairLabel), 1.);
					wordPairFv.add(candidateLabel.concat(wBase).concat(pairLabel), 1.);
					if(params.getUseFullFeatures()){
						// w-rel-w
						int w2Idx = ((JointInstance)instance).getWord(pair.getW2().getId());
						if(w2Idx <= lastIndex){
							String w2Base = pair.getW2().getWord().getRealBase();
							String w2Label = ((WordLabelUnit)y.getLabel(w2Idx)).getLabel();
							wordPairFv.add(candidateLabel.concat(pairLabel).concat(w2Label), 1.);
							wordPairFv.add(candidateLabel.concat(wBase).concat(pairLabel).concat(w2Label).concat(w2Base), 1.);
						}
					}
				}else if(pair.getW2().getId() == word.getId()){
					String pairLabel = ((PairLabelUnit)y.getLabel(i)).getLabel();
					wordPairFv.add(pairLabel.concat(candidateLabel), 1.);
					wordPairFv.add(pairLabel.concat(wBase).concat(candidateLabel), 1.);
					if(params.getUseFullFeatures()){
						// w-rel-w
						int w1Idx = ((JointInstance)instance).getWord(pair.getW1().getId());
						if(w1Idx <= lastIndex){
							String w1Base = pair.getW1().getWord().getRealBase();
							String w1Label = ((WordLabelUnit)y.getLabel(w1Idx)).getLabel();
							wordPairFv.add(w1Label.concat(pairLabel).concat(candidateLabel), 1.);
							wordPairFv.add(w1Base.concat(w1Label).concat(pairLabel).concat(candidateLabel).concat(wBase), 1.);
						}
					}
				}
			}else{
				assert y.getLabel(i) instanceof WordLabelUnit;
				if(!params.getUseGlobalEntityFeatures())continue;
				Word w2 = (Word)instance.getSequence().get(i);
				if(w2.getId() == word.getId() - 2){
					contextWords[0] = i;
				}if(w2.getId() == word.getId() - 1){
					String wordLabel = ((WordLabelUnit)y.getLabel(i)).getLabel();
					contextWords[1] = i;
					addBigramToFV(wordFv, wordLabel, w2.getWord().getRealBase(), w2.getWord().getPOS(), candidateLabel, wBase, wPOS);				
				}else if(word.getId() == w2.getId() - 1){
					String wordLabel = ((WordLabelUnit)y.getLabel(i)).getLabel();
					contextWords[2] = i;
					addBigramToFV(wordFv, candidateLabel, wBase, wPOS, wordLabel, w2.getWord().getRealBase(), w2.getWord().getPOS());
				}else if(word.getId() == w2.getId() - 2){
					contextWords[3] = i;
				}
			}
		}
		
		List<Word> entityWords = getEntityWords(word, instance, y, lastIndex, candidateLabelUnit);
		if(entityWords.size() > 0){
			if(params.getUseFullFeatures()){
				String entString = entityString(entityWords);
				String candidateLabelType = candidateLabelUnit.getType();
				wordFv.add(candidateLabelType.concat(entString), 1.);
				Word lastWord = entityWords.get(entityWords.size() - 1);
				for(int i = 0;i <= lastIndex;i++){
					if(y.getLabel(i) instanceof PairLabelUnit){
						if(y.getLabel(i).isNegative())continue;
						Pair pair = (Pair)instance.getSequence().get(i);
						if(pair.getW1().getId() == lastWord.getId()){
							String pairLabel = ((PairLabelUnit)y.getLabel(i)).getLabel();
							wordPairFv.add(candidateLabelType.concat(pairLabel), 1.);
							wordPairFv.add(candidateLabelType.concat(entString).concat(pairLabel), 1.);
							// ent-rel-ent
							int w2Idx = ((JointInstance)instance).getWord(pair.getW2().getId());
							if(w2Idx <= lastIndex){
								List<Word> e2Words = getEntityWords(pair.getW2(), instance, y, lastIndex, (WordLabelUnit)y.getLabel(w2Idx));
								if(e2Words.size() > 0){
									String e2String = entityString(e2Words);
									String w2LabelType = ((WordLabelUnit)y.getLabel(w2Idx)).getType();
									wordPairFv.add(candidateLabelType.concat(pairLabel).concat(w2LabelType), 1.);
									wordPairFv.add(candidateLabelType.concat(entString).concat(pairLabel).concat(w2LabelType).concat(e2String), 1.);
								}
							}
						}else if(pair.getW2().getId() == lastWord.getId()){
							String pairLabel = ((PairLabelUnit)y.getLabel(i)).getLabel();
							wordPairFv.add(pairLabel.concat(candidateLabelType), 1.);
							wordPairFv.add(pairLabel.concat(candidateLabelType).concat(entString), 1.);
							// ent-rel-ent
							int w1Idx = ((JointInstance)instance).getWord(pair.getW1().getId());
							if(w1Idx <= lastIndex){
								List<Word> e1Words = getEntityWords(pair.getW1(), instance, y, lastIndex, (WordLabelUnit)y.getLabel(w1Idx));
								if(e1Words.size() > 0){
									String e1String = entityString(e1Words);
									String w1LabelType = ((WordLabelUnit)y.getLabel(w1Idx)).getType();
									wordPairFv.add(w1LabelType.concat(pairLabel).concat(candidateLabelType), 1.);
									wordPairFv.add(w1LabelType.concat(e1String).concat(pairLabel).concat(candidateLabelType).concat(entString), 1.);
								}
							}
						}
					}
				}
			}
		}		
		if(contextWords[0] >= 0 && contextWords[1] >= 0){
			Word w0 = (Word)instance.getSequence().get(contextWords[0]);
			String l0 = ((WordLabelUnit)y.getLabel(contextWords[0])).getLabel();
			Word w1 = (Word)instance.getSequence().get(contextWords[1]);
			String l1 = ((WordLabelUnit)y.getLabel(contextWords[1])).getLabel();
			addTrigramToFV(wordFv, l0, w0.getWord().getRealBase(), w0.getWord().getPOS(), l1, w1.getWord().getRealBase(), w1.getWord().getPOS(), candidateLabel, wBase, wPOS);			
		}
		if(contextWords[1] >= 0 && contextWords[2] >= 0){
			Word w1 = (Word)instance.getSequence().get(contextWords[1]);
			String l1 = ((WordLabelUnit)y.getLabel(contextWords[1])).getLabel();
			Word w2 = (Word)instance.getSequence().get(contextWords[2]);
			String l2 = ((WordLabelUnit)y.getLabel(contextWords[2])).getLabel();
			addTrigramToFV(wordFv, l1, w1.getWord().getRealBase(), w1.getWord().getPOS(), candidateLabel, wBase, wPOS, l2, w2.getWord().getRealBase(), w2.getWord().getPOS());	
		}
		if(contextWords[2] >= 0 && contextWords[3] >= 0){
			Word w2 = (Word)instance.getSequence().get(contextWords[2]);
			String l2 = ((WordLabelUnit)y.getLabel(contextWords[2])).getLabel();
			Word w3 = (Word)instance.getSequence().get(contextWords[3]);
			String l3 = ((WordLabelUnit)y.getLabel(contextWords[3])).getLabel();
			addTrigramToFV(wordFv, candidateLabel, wBase, wPOS, l2, w2.getWord().getRealBase(), w2.getWord().getPOS(), l3, w3.getWord().getRealBase(), w3.getWord().getPOS());
		}	
		globalFv.add(wordFv, "WORD");
		wordPairFv.mult(params.getRelWeight());
		globalFv.add(wordPairFv, "WORDPAIR");
		return globalFv;		
	}
	
	protected void addSecondOrderInfoToFV(StringSparseVector fv, String label1, String base1, String pos1, String label2, String header){
		StringSparseVector secondOrderInfoFv = new StringSparseVector(params);
		secondOrderInfoFv.add(header.concat(label1).concat(base1).concat(label2), 1.);
		secondOrderInfoFv.add(header.concat(label1).concat(pos1).concat(label2), 1.);
		secondOrderInfoFv.add(header.concat(label1).concat(label2), 1.);
		secondOrderInfoFv.normalize(1.);
		fv.add(secondOrderInfoFv);
	}
	

	protected void addTriangleToFV(StringSparseVector fv, String label1, String base1, String pos1, String label2, String base2, String pos2, String label3, String base3, String pos3){
		StringSparseVector triangleFv = new StringSparseVector(params);
		triangleFv.add(label1.concat(label2).concat(label3).concat("TRIANGLE"), 1.);
		triangleFv.add(label1.concat(base1).concat(label2).concat(label3).concat("TRIANGLE"), 1.);
		triangleFv.add(label1.concat(label2).concat(base2).concat(label3).concat("TRIANGLE"), 1.);
		triangleFv.add(label1.concat(label2).concat(label3).concat(base3).concat("TRIANGLE"), 1.);
		triangleFv.add(label1.concat(pos1).concat(label2).concat(label3).concat("TRIANGLE"), 1.);
		triangleFv.add(label1.concat(label2).concat(pos2).concat(label3).concat("TRIANGLE"), 1.);
		triangleFv.add(label1.concat(label2).concat(label3).concat(pos3).concat("TRIANGLE"), 1.);
		triangleFv.add("TRIANGLE", 1.);
		triangleFv.normalize(1.);
		fv.add(triangleFv);
	}
	

	protected void addParallelInfoToFV(SparseFeatureVector fv, String wBase, String wPOS, Node w1, String label1, Node w2, String label2, String header){
		SparseFeatureVector paraFv = new SparseFeatureVector(params);
		paraFv.add(w1.getCache().getShortestPathFeatures(w2), header.concat(label1).concat(wBase).concat(label2));
		paraFv.add(w1.getCache().getShortestPathFeatures(w2), header.concat(label1).concat(wPOS).concat(label2));
		paraFv.add(w1.getCache().getShortestPathFeatures(w2), header.concat(label1).concat(label2));
		paraFv.normalize(1.);
		fv.add(paraFv);
	}
	
	protected SparseFeatureVector calcPairGlobalFeatures(Pair pair, Instance instance, Label y, int lastIndex, PairLabelUnit candidateLabelUnit){
		SparseFeatureVector globalFv = new SparseFeatureVector(params);
		if(candidateLabelUnit.isNegative()){
			return globalFv;
		}
		SparseFeatureVector parallelFv = new SparseFeatureVector(params);		
		StringSparseVector counterFv = new StringSparseVector(params);
		StringSparseVector wordPairFv = new StringSparseVector(params);

		String candidateLabel = candidateLabelUnit.getLabel();
		Word w1 = pair.getW1();
		Word w2 = pair.getW2();		

		JointInstance jInstance = (JointInstance)instance;
		int w1Index = jInstance.getWord(w1.getId());
		int w2Index = jInstance.getWord(w2.getId());

		String w1Base = w1.getWord().getRealBase();
		String w1POS = w1.getWord().getPOS();
		String w2Base = w2.getWord().getRealBase();
		String w2POS = w2.getWord().getPOS();

		for(int i = 0;i <= lastIndex;i++){
			if(y.getLabel(i).isNegative())continue;
			if(y.getLabel(i) instanceof PairLabelUnit){
				if(!params.getUseGlobalRelationFeatures())continue;
				PairLabelUnit adjLabelUnit = (PairLabelUnit)y.getLabel(i);
				String adjLabel = adjLabelUnit.getLabel();
				Pair adjPair = (Pair)instance.getSequence().get(i);
				Word adjW1 = adjPair.getW1();
				Word adjW2 = adjPair.getW2();
				if(adjW1.getId() == w1.getId()){
					if(adjW2.getId() < w2.getId()){
						addSecondOrderInfoToFV(counterFv, candidateLabel, w1Base, w1POS, adjLabel, "PARA-E1");
						addParallelInfoToFV(parallelFv, w1Base, w1POS, adjW2.getWord(), adjLabel, w2.getWord(), candidateLabel, "PARA-E1");
						int triangleIndex = jInstance.getWordPair(adjW2.getId(), w2.getId());
						if(triangleIndex <= lastIndex){
							PairLabelUnit thirdLabelUnit = ((PairLabelUnit)y.getLabel(triangleIndex));
							if(!thirdLabelUnit.isNegative()){
								addTriangleToFV(counterFv, adjLabel, w1Base, w1POS, candidateLabel, w2Base, w2POS, thirdLabelUnit.getLabel(), adjW2.getWord().getRealBase(), adjW2.getWord().getPOS());
							}
						}
					}else{
						addSecondOrderInfoToFV(counterFv, adjLabel, w1Base, w1POS, candidateLabel, "PARA-E1");
						addParallelInfoToFV(parallelFv, w1Base, w1POS, w2.getWord(), candidateLabel, adjW2.getWord(), adjLabel, "PARA-E1");
						int triangleIndex = jInstance.getWordPair(w2.getId(), adjW2.getId());
						if(triangleIndex <= lastIndex){
							PairLabelUnit thirdLabelUnit = ((PairLabelUnit)y.getLabel(triangleIndex));
							if(!thirdLabelUnit.isNegative()){
								addTriangleToFV(counterFv, candidateLabel, w1Base, w1POS, adjLabel, adjW2.getWord().getRealBase(), adjW2.getWord().getPOS(), thirdLabelUnit.getLabel(), w2Base, w2POS);
							}
						}
					}
				}else if(adjW2.getId() == w1.getId()){
					if(adjW1.getId() < w2.getId()){
						addParallelInfoToFV(parallelFv, w1Base, w1POS, adjW1.getWord(), adjLabel, w2.getWord(), candidateLabel, "SEQ");
						int triangleIndex = jInstance.getWordPair(adjW1.getId(), w2.getId());
						if(triangleIndex <= lastIndex){
							PairLabelUnit thirdLabelUnit = ((PairLabelUnit)y.getLabel(triangleIndex));
							if(!thirdLabelUnit.isNegative()){
								addTriangleToFV(counterFv, adjLabel, w1Base, w1POS, candidateLabel, w2Base, w2POS, thirdLabelUnit.getLabel(), adjW1.getWord().getRealBase(), adjW1.getWord().getPOS());
							}
						}
					}else{
						addParallelInfoToFV(parallelFv, w1Base, w1POS, w2.getWord(), adjLabel, adjW1.getWord(), candidateLabel, "RSEQ");
						int triangleIndex = jInstance.getWordPair(w2.getId(), adjW1.getId());
						if(triangleIndex <= lastIndex){
							PairLabelUnit thirdLabelUnit = ((PairLabelUnit)y.getLabel(triangleIndex));
							if(!thirdLabelUnit.isNegative()){
								addTriangleToFV(counterFv, candidateLabel, w1Base, w1POS, adjLabel, adjW1.getWord().getRealBase(), adjW1.getWord().getPOS(), thirdLabelUnit.getLabel(), w2Base, w2POS);
							}
						}
					}
					addSecondOrderInfoToFV(counterFv, candidateLabel, w1Base, w1POS, adjLabel, "SEQ");
				}else if(adjW2.getId() == w2.getId()){
					if(adjW1.getId() < w1.getId()){
						addParallelInfoToFV(parallelFv, w2Base, w2POS, adjW1.getWord(), adjLabel, w1.getWord(), candidateLabel, "PARA-E2");
						addSecondOrderInfoToFV(counterFv, candidateLabel, w2Base, w2POS, adjLabel, "PARA-E2");
						int triangleIndex = jInstance.getWordPair(adjW1.getId(), w1.getId());
						if(triangleIndex <= lastIndex){
							PairLabelUnit thirdLabelUnit = ((PairLabelUnit)y.getLabel(triangleIndex));
							if(!thirdLabelUnit.isNegative()){
								addTriangleToFV(counterFv, adjLabel, w2Base, w2POS, candidateLabel, w1Base, w1POS, thirdLabelUnit.getLabel(), adjW1.getWord().getRealBase(), adjW1.getWord().getPOS());
							}
						}
					}else{
						addParallelInfoToFV(parallelFv, w2Base, w2POS, w1.getWord(), candidateLabel, adjW1.getWord(), adjLabel, "PARA-E2");
						addSecondOrderInfoToFV(counterFv, adjLabel, w2Base, w2POS, candidateLabel, "PARA-E2");
						int triangleIndex = jInstance.getWordPair(w1.getId(), adjW1.getId());
						if(triangleIndex <= lastIndex){
							PairLabelUnit thirdLabelUnit = ((PairLabelUnit)y.getLabel(triangleIndex));
							if(!thirdLabelUnit.isNegative()){
								addTriangleToFV(counterFv, candidateLabel, w2Base, w2POS, adjLabel, adjW1.getWord().getRealBase(), adjW1.getWord().getPOS(), thirdLabelUnit.getLabel(), w1Base, w1POS);
							}
						}
					}
				}else if(adjW1.getId() == w2.getId()){
					if(w1.getId() < adjW2.getId()){
						addParallelInfoToFV(parallelFv, w2Base, w2POS, w1.getWord(), candidateLabel, adjW2.getWord(), adjLabel, "SEQ");
						int triangleIndex = jInstance.getWordPair(w1.getId(), adjW2.getId());
						if(triangleIndex <= lastIndex){
							PairLabelUnit thirdLabelUnit = ((PairLabelUnit)y.getLabel(triangleIndex));
							if(!thirdLabelUnit.isNegative()){
								addTriangleToFV(counterFv, candidateLabel, w2Base, w2POS, adjLabel, adjW2.getWord().getRealBase(), adjW2.getWord().getPOS(), thirdLabelUnit.getLabel(), w1Base, w1POS);
							}
						}
					}else{
						addParallelInfoToFV(parallelFv, w2Base, w2POS, adjW2.getWord(), candidateLabel, w1.getWord(), adjLabel, "RSEQ");
						int triangleIndex = jInstance.getWordPair(adjW2.getId(), w1.getId());
						if(triangleIndex <= lastIndex){
							PairLabelUnit thirdLabelUnit = ((PairLabelUnit)y.getLabel(triangleIndex));
							if(!thirdLabelUnit.isNegative()){
								addTriangleToFV(counterFv, adjLabel, w2Base, w2POS, candidateLabel, w1Base, w1POS, thirdLabelUnit.getLabel(), adjW2.getWord().getRealBase(), adjW2.getWord().getPOS());
							}
						}
					}
					addSecondOrderInfoToFV(counterFv, adjLabel, w2Base, w2POS, candidateLabel, "SEQ");
				}else{
					if(adjW1.getId() < w1.getId() && w1.getId() < adjW2.getId() && adjW2.getId() < w2.getId()){
						counterFv.add("PROJ".concat(adjLabel).concat(candidateLabel), 1.);
					}else if(w1.getId() < adjW1.getId() && adjW1.getId() < w2.getId() && w2.getId() < adjW2.getId()){
						counterFv.add("PROJ".concat(candidateLabel).concat(adjLabel), 1.);
					}
				}
			}else{
				if(w1Index == i){
					assert y.getLabel(i) instanceof WordLabelUnit;
					String w1Label = ((WordLabelUnit)y.getLabel(i)).getLabel();
					wordPairFv.add(w1Label.concat(candidateLabel), 1.);
					wordPairFv.add(w1Label.concat(w1Base).concat(candidateLabel), 1.);
				}
				if(w2Index == i){
					assert y.getLabel(i) instanceof WordLabelUnit;
					String w2Label = ((WordLabelUnit)y.getLabel(i)).getLabel();
					wordPairFv.add(candidateLabel.concat(w2Label), 1.);
					wordPairFv.add(candidateLabel.concat(w2Base).concat(w2Label), 1.);
				}
			}
		}
		if(params.getUseFullFeatures()){
			int w1Idx = ((JointInstance)instance).getWord(pair.getW1().getId());
			int w2Idx = ((JointInstance)instance).getWord(pair.getW2().getId());
			if(w1Idx <= lastIndex && w2Idx <= lastIndex){
				// w-rel-w
				String w1Label = ((WordLabelUnit)y.getLabel(w1Idx)).getLabel();
				String w2Label = ((WordLabelUnit)y.getLabel(w2Idx)).getLabel();
				wordPairFv.add(w1Label.concat(candidateLabel).concat(w2Label), 1.);
				wordPairFv.add(w1Label.concat(w1Base).concat(candidateLabel).concat(w2Label).concat(w2Base), 1.);
				List<Word> e1Words = getEntityWords(pair.getW1(), instance, y, lastIndex, (WordLabelUnit)y.getLabel(w1Idx));
				String e1String = "", e2String = "", w1LabelType = "", w2LabelType = "";
				if(e1Words.size() != 0){
					e1String = entityString(e1Words);
					w1LabelType = ((WordLabelUnit)y.getLabel(w1Idx)).getType();
					wordPairFv.add(w1LabelType.concat(candidateLabel), 1.);
					wordPairFv.add(w1LabelType.concat(e1String).concat(candidateLabel), 1.);
				}
				List<Word> e2Words = getEntityWords(pair.getW2(), instance, y, lastIndex, (WordLabelUnit)y.getLabel(w2Idx));
				if(e2Words.size() != 0){
					e2String = entityString(e2Words);
					w2LabelType = ((WordLabelUnit)y.getLabel(w2Idx)).getType();
					wordPairFv.add(candidateLabel.concat(w2LabelType), 1.);
					wordPairFv.add(candidateLabel.concat(w2LabelType).concat(e2String), 1.);
				}
				// ent-rel-ent
				if(e1Words.size() != 0 && e2Words.size() != 0){	
					wordPairFv.add(w1LabelType.concat(candidateLabel).concat(w2LabelType), 1.);
					wordPairFv.add(w1LabelType.concat(e1String).concat(candidateLabel).concat(w2LabelType).concat(e2String), 1.);
				}
			}
		}
		globalFv.add(parallelFv, "PARA");
		globalFv.add(counterFv, "COUNT");
		globalFv.add(wordPairFv, "WORDPAIR");
		globalFv.scale(params.getRelWeight());
		return globalFv;		
	}

}
