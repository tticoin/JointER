package data.nlp.joint;

import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.Vector;
import java.util.Map.Entry;

import com.google.common.collect.Lists;

import config.Parameters;
import data.Instance;
import data.Label;
import data.LabelUnit;
import data.Sequence;
import data.nlp.Document;
import data.nlp.Node;
import data.nlp.Offset;
import data.nlp.Sentence;
import data.nlp.relation.Relation;

public class JointDocument extends Document {
	protected List<Relation> relations;

	public JointDocument(Parameters params, String id) {
		this(params, null, null, id, "", "");
	}
	
	public JointDocument(Parameters params, Document document, Offset offset, String id, String type, String source) {
		super(params, document, offset, id, type, source);
		this.relations = new Vector<Relation>();
	}
	
	
	public void buildRelations(boolean isTrain) {
		Map<String, Node> goldEntities = new TreeMap<String, Node>(); 
		for(Sentence sentence:sentences){
			goldEntities.putAll(sentence.getGoldEntities());
		}
		for(Sentence sentence:sentences){
			sentence.processParse();
		}
		Set<String> relationKeys = new TreeSet<String>();
		for(Relation relation:relations){
			String type = relation.getType();
			for(Entry<String, String> arg1:relation.getArguments().entries()){
				relationKeys.add(arg1.getKey());
				assert goldEntities.containsKey(arg1.getValue()): arg1.getValue();
				Node arg1Node = goldEntities.get(arg1.getValue());
				for(Entry<String, String> arg2:relation.getArguments().entries()){
					if(arg1.getKey().equals(arg2.getKey()))continue;
					assert goldEntities.containsKey(arg2.getValue()): id+":"+arg2.getValue();
					Node arg2Node = goldEntities.get(arg2.getValue());
					if(arg1Node.getNodes().getSentence() != arg2Node.getNodes().getSentence())continue;
					if(arg1Node.getOffset().compareTo(arg2Node.getOffset()) > 0)continue;
					if(arg1Node.getOffset().compareTo(arg2Node.getOffset()) == 0){
						if(arg1Node.getId().compareTo(arg2Node.getId()) < 0){
							continue;
						}
					}
					if(params.getUseDirectedRelation()){
						arg1Node.addGoldRelation(arg2Node, arg1.getKey()+":"+type+":"+arg2.getKey());
						arg2Node.addGoldRelation(arg1Node, arg2.getKey()+":"+type+":"+arg1.getKey());
					}else{
						arg1Node.addGoldRelation(arg2Node, type);
						arg2Node.addGoldRelation(arg1Node, type);
					}
				}				
			}
		}
		assert relations.size() == 0 || relationKeys.size() == 2: relationKeys;
		String[] relArgs = relationKeys.toArray(new String[2]);
		assert relations.size() == 0 || relArgs.length == 2: relArgs;

		relations.clear();
		for(Sentence sentence:sentences){
			//sentence
			Set<Node> entities = new TreeSet<Node>();
			entities.addAll(sentence.getGoldEntities().values());
			// entity -> words
			Map<Node, List<Node>> entityWordsMap = new TreeMap<Node, List<Node>>();
			Iterator<Node> nodeIt = entities.iterator();
			Node node = nodeIt.hasNext() ? nodeIt.next() : null;
			for(Node word:sentence.getWords(params.getSentenceAnnotation())){
				if(node == null)continue;
				if(word.isInsideOf(node)){
					if(!entityWordsMap.containsKey(node)){
						entityWordsMap.put(node, new Vector<Node>());
					}
					entityWordsMap.get(node).add(word);
				}else if(word.getOffset().getStart() >= node.getOffset().getEnd()){
					if(!entityWordsMap.containsKey(node)){
						Node headWord = node.getFirstHeadWord(params.getSentenceAnnotation());	
						boolean isDuplicated = false;
						for(Node entity:entityWordsMap.keySet()){
							if(entityWordsMap.get(entity).contains(headWord)){
								if(entityWordsMap.get(entity).size() > 1){
									entityWordsMap.get(entity).remove(headWord);
								}else{
									if(params.getVerbosity() > 0){
										System.err.print(headWord.getText()+" has duplicated entities in "+sentence.getNode().getText()+". Ignore ");
										if(entity.getOffset().getEnd() < node.getOffset().getEnd()){
											System.err.println(entity.getText());
										}else{
											System.err.println(node.getText());
										}
									}
									if(entity.getOffset().getEnd() < node.getOffset().getEnd()){
										entityWordsMap.remove(entity); 
									}else{
										isDuplicated = true;	
									}
									break;
									//assert false: headWord.getText()+":"+node.getText()+":"+sentence.getNode().getText();
								}
							}
						}						
						if(!isDuplicated){
							entityWordsMap.put(node, new Vector<Node>());
							entityWordsMap.get(node).add(headWord);
						}
					}
					node = nodeIt.hasNext() ? nodeIt.next() : null;
					if(node == null)continue;
					if(word.isInsideOf(node)){
						if(!entityWordsMap.containsKey(node)){
							entityWordsMap.put(node, new Vector<Node>());
						}
						entityWordsMap.get(node).add(word);						
					}else{
						//assert word.getOffset().getEnd() <= node.getOffset().getEnd(): word.getText()+":"+node.getText()+":"+sentence.getNode().getText();
					}
				}
			}
			assert !nodeIt.hasNext(): this.id+":"+nodeIt.next().getId();
			//assert entityWordsMap.size() == entities.size(): entityWordsMap.size()+":"+entities.size();
			// word -> label, word -> entity
			Map<Node, LabelUnit> goldLabels = new TreeMap<Node, LabelUnit>();
			Map<Node, Node> wordEntityMap = new TreeMap<Node, Node>();
			for(Node entity:entityWordsMap.keySet()){
				String type = entity.getType();
				List<Node> words = entityWordsMap.get(entity);
				if(words.size() == 1){
					assert !goldLabels.containsKey(words.get(0));
					goldLabels.put(words.get(0), new WordLabelUnit(WordLabelUnit.U, type));
					wordEntityMap.put(words.get(0), entity);
				}else{
					int nwords = words.size();
					for(int index = 0;index < nwords;index++){
						assert !goldLabels.containsKey(words.get(index));
						if(index == 0){
							goldLabels.put(words.get(index), new WordLabelUnit(WordLabelUnit.B, type));
						}else if(index == nwords - 1){
							goldLabels.put(words.get(index), new WordLabelUnit(WordLabelUnit.L, type));
							wordEntityMap.put(words.get(index), entity);
						}else{
							goldLabels.put(words.get(index), new WordLabelUnit(WordLabelUnit.I, type));
						}
					}
				}
			}
			for(Node word:sentence.getWords(params.getSentenceAnnotation())){
				if(!goldLabels.containsKey(word)){
					goldLabels.put(word, WordLabelUnit.getNegativeClassLabelUnit());
				}
			}
			Label label = new Label(params);
			JointSequence seq = new JointSequence(params);
			if(params.useLtoR()){
				createLtoRSequence(seq, label, sentence, goldLabels, wordEntityMap);
			}else if(params.useRtoL()){
				createRtoLSequence(seq, label, sentence, goldLabels, wordEntityMap);
			}else if(params.useEntityLtoR()){
				createEntityLtoRSequence(seq, label, sentence, goldLabels, wordEntityMap);
			}else if(params.useEntityRtoL()){
				createEntityRtoLSequence(seq, label, sentence, goldLabels, wordEntityMap);
			}else if(params.useCloseFirst() || params.useSort()){
				createCloseFirstLtoRSequence(seq, label, sentence, goldLabels, wordEntityMap);
			}else if(params.useCloseFirstRtoL()){
				createCloseFirstRtoLSequence(seq, label, sentence, goldLabels, wordEntityMap);
			}
			//TODO: check no word sentence??
			if(seq.size() == 0)continue;
			Instance instance = createInstance(params, seq, label);
			assert label.size() == seq.size();
			assert label.size() == sentence.getWords(params.getSentenceAnnotation()).size() * (sentence.getWords(params.getSentenceAnnotation()).size() + 3) / 2 - (params.getUseSelfRelation() ? 0 : sentence.getWords(params.getSentenceAnnotation()).size()) : label.size()+":"+sentence.getWords(params.getSentenceAnnotation()).size();			
			instances.add(instance);
		}
	}

	protected Instance createInstance(Parameters params, Sequence seq, Label label){
		return new JointInstance(params, seq, label);
	}

	private void createLtoRSequence(JointSequence seq, Label label, Sentence sentence, Map<Node, LabelUnit> goldLabels, Map<Node, Node> wordEntityMap){
		List<Node> words = sentence.getWords(params.getSentenceAnnotation());
		Map<Integer, Word> wordMap = new TreeMap<Integer, Word>();
		for(int w1Index = 0; w1Index < words.size();w1Index++){
			Node w1 = words.get(w1Index);
			Word w1Word = new Word(params, w1, w1Index);
			wordMap.put(w1Index, w1Word);
			for(int w2Index = 0; w2Index <= w1Index;w2Index++){
				Node w2 = words.get(w2Index);
				if(!params.getUseSelfRelation() && w1 == w2){
					continue;
				}
				assert w2.getOffset().compareTo(w1.getOffset()) <= 0:w2.getOffset()+":"+w1.getOffset();
				if(!wordEntityMap.containsKey(w1) || !wordEntityMap.containsKey(w2)){
					label.add(PairLabelUnit.getNegativeClassLabelUnit());
				}else{
					Collection<String> rels = wordEntityMap.get(w2).getRelations().get(wordEntityMap.get(w1));
					if(rels.size() == 0){
						label.add(PairLabelUnit.getNegativeClassLabelUnit());
					}else{
						label.add(new PairLabelUnit(rels));
					}
				}
				assert wordMap.containsKey(w2Index);
				seq.add(new Pair(params, wordMap.get(w2Index), w1Word));					
			}
			label.add(goldLabels.get(w1));
			seq.add(w1Word);
		}
	}

	private void createRtoLSequence(JointSequence seq, Label label, Sentence sentence, Map<Node, LabelUnit> goldLabels, Map<Node, Node> wordEntityMap){
		List<Node> reverseWords = Lists.reverse(sentence.getWords(params.getSentenceAnnotation()));
		Map<Integer, Word> wordMap = new TreeMap<Integer, Word>();
		int nwords = reverseWords.size();
		for(int w1Index = 0; w1Index < reverseWords.size();w1Index++){
			Node w1 = reverseWords.get(w1Index);
			Word w1Word = new Word(params, w1, nwords - 1 - w1Index);
			wordMap.put(w1Index, w1Word);
			for(int w2Index = 0; w2Index <= w1Index;w2Index++){
				Node w2 = reverseWords.get(w2Index);
				if(!params.getUseSelfRelation() && w1 == w2){
					continue;
				}
				assert w2.getOffset().compareTo(w1.getOffset()) >= 0:w2.getOffset()+":"+w1.getOffset();
				if(!wordEntityMap.containsKey(w1) || !wordEntityMap.containsKey(w2)){
					label.add(PairLabelUnit.getNegativeClassLabelUnit());
				}else{
					Collection<String> rels = wordEntityMap.get(w1).getRelations().get(wordEntityMap.get(w2));
					if(rels.size() == 0){
						label.add(PairLabelUnit.getNegativeClassLabelUnit());
					}else{
						label.add(new PairLabelUnit(rels));
					}
				}
				seq.add(new Pair(params, w1Word, wordMap.get(w2Index)));					
			}
			label.add(goldLabels.get(w1));
			seq.add(w1Word);
		}	
	}
	
	private void createEntityLtoRSequence(JointSequence seq, Label label, Sentence sentence, Map<Node, LabelUnit> goldLabels, Map<Node, Node> wordEntityMap){
		List<Node> words = sentence.getWords(params.getSentenceAnnotation());
		Map<Integer, Word> wordMap = new TreeMap<Integer, Word>();
		for(int w1Index = 0; w1Index < words.size();w1Index++){
			Node w1 = words.get(w1Index);
			Word w1Word = new Word(params, w1, w1Index);
			wordMap.put(w1Index, w1Word);
			label.add(goldLabels.get(w1));
			seq.add(w1Word);
			for(int w2Index = w1Index; w2Index >= 0;w2Index--){
				Node w2 = words.get(w2Index);
				if(!params.getUseSelfRelation() && w1 == w2){
					continue;
				}
				assert w2.getOffset().compareTo(w1.getOffset()) <= 0:w2.getOffset()+":"+w1.getOffset();
				if(!wordEntityMap.containsKey(w1) || !wordEntityMap.containsKey(w2)){
					label.add(PairLabelUnit.getNegativeClassLabelUnit());
				}else{
					Collection<String> rels = wordEntityMap.get(w2).getRelations().get(wordEntityMap.get(w1));
					if(rels.size() == 0){
						label.add(PairLabelUnit.getNegativeClassLabelUnit());
					}else{
						label.add(new PairLabelUnit(rels));
					}
				}
				assert wordMap.containsKey(w2Index);
				seq.add(new Pair(params, wordMap.get(w2Index), w1Word));					
			}
		}
	}

	private void createEntityRtoLSequence(JointSequence seq, Label label, Sentence sentence, Map<Node, LabelUnit> goldLabels, Map<Node, Node> wordEntityMap){
		List<Node> reverseWords = Lists.reverse(sentence.getWords(params.getSentenceAnnotation()));
		Map<Integer, Word> wordMap = new TreeMap<Integer, Word>();
		int nwords = reverseWords.size();
		for(int w1Index = 0; w1Index < reverseWords.size();w1Index++){
			Node w1 = reverseWords.get(w1Index);
			Word w1Word = new Word(params, w1, nwords - 1 - w1Index);
			wordMap.put(w1Index, w1Word);
			label.add(goldLabels.get(w1));
			seq.add(w1Word);
			for(int w2Index = w1Index; w2Index >= 0;w2Index--){
				Node w2 = reverseWords.get(w2Index);
				if(!params.getUseSelfRelation() && w1 == w2){
					continue;
				}
				assert w2.getOffset().compareTo(w1.getOffset()) >= 0:w2.getOffset()+":"+w1.getOffset();
				if(!wordEntityMap.containsKey(w1) || !wordEntityMap.containsKey(w2)){
					label.add(PairLabelUnit.getNegativeClassLabelUnit());
				}else{
					Collection<String> rels = wordEntityMap.get(w1).getRelations().get(wordEntityMap.get(w2));
					if(rels.size() == 0){
						label.add(PairLabelUnit.getNegativeClassLabelUnit());
					}else{
						label.add(new PairLabelUnit(rels));
					}
				}
				seq.add(new Pair(params, w1Word, wordMap.get(w2Index)));					
			}
		}	
	}

	

	private void createCloseFirstLtoRSequence(JointSequence seq, Label label, Sentence sentence, Map<Node, LabelUnit> goldLabels, Map<Node, Node> wordEntityMap){
		List<Node> words = sentence.getWords(params.getSentenceAnnotation());
		int index = 0;
		for(Node word:sentence.getWords(params.getSentenceAnnotation())){
			label.add(goldLabels.get(word));
			seq.add(new Word(params, word, index));
			index++;
		}
		for(int i = 0;i < words.size();i++){
			if(!params.getUseSelfRelation() && i == 0){
				continue;
			}
			for(int w1Index = 0; w1Index < words.size() - i;w1Index++){
				Node w1 = words.get(w1Index);
				int w2Index = w1Index + i;
				Node w2 = words.get(w1Index+i);
				assert w1.getOffset().compareTo(w2.getOffset()) <= 0:w1.getOffset()+":"+w2.getOffset();
				if(!wordEntityMap.containsKey(w1) || !wordEntityMap.containsKey(w2)){
					label.add(PairLabelUnit.getNegativeClassLabelUnit());
				}else{
					Collection<String> rels = wordEntityMap.get(w1).getRelations().get(wordEntityMap.get(w2));
					if(rels.size() == 0){
						label.add(PairLabelUnit.getNegativeClassLabelUnit());
					}else{
						label.add(new PairLabelUnit(rels));
					}
				}
				assert ((Word)seq.get(w2Index)).getId() == w2Index;
				assert ((Word)seq.get(w1Index)).getId() < ((Word)seq.get(w2Index)).getId();
				seq.add(new Pair(params, (Word)seq.get(w1Index), (Word)seq.get(w2Index)));					
			}		
		}		
	}

	private void createCloseFirstRtoLSequence(JointSequence seq, Label label, Sentence sentence, Map<Node, LabelUnit> goldLabels, Map<Node, Node> wordEntityMap){
		List<Node> reverseWords = Lists.reverse(sentence.getWords(params.getSentenceAnnotation()));
		int index = reverseWords.size() - 1;
		for(Node word:reverseWords){
			label.add(goldLabels.get(word));
			seq.add(new Word(params, word, index));
			index--;
		}
		assert index == -1: index;

		for(int i = 0;i < reverseWords.size();i++){
			for(int w1Index = 0; w1Index < reverseWords.size() - i;w1Index++){
				Node w1 = reverseWords.get(w1Index);
				int w2Index = w1Index + i;
				Node w2 = reverseWords.get(w2Index);
				if(!params.getUseSelfRelation() && w1 == w2){
					continue;
				}
				assert w2.getOffset().compareTo(w1.getOffset()) <= 0:w2.getOffset()+":"+w1.getOffset();
				if(!wordEntityMap.containsKey(w1) || !wordEntityMap.containsKey(w2)){
					label.add(PairLabelUnit.getNegativeClassLabelUnit());
				}else{
					Collection<String> rels = wordEntityMap.get(w2).getRelations().get(wordEntityMap.get(w1));
					if(rels.size() == 0){
						label.add(PairLabelUnit.getNegativeClassLabelUnit());
					}else{
						label.add(new PairLabelUnit(rels));
					}
				}
				assert ((Word)seq.get(w2Index)).getId() == reverseWords.size() - 1 - w2Index;
				assert ((Word)seq.get(w2Index)).getId() < ((Word)seq.get(w1Index)).getId();
				seq.add(new Pair(params, (Word)seq.get(w2Index), (Word)seq.get(w1Index)));					
			}
		}			
	}
	
	public void addRelation(Relation relation) {
		this.relations.add(relation);	
	}
	
}
