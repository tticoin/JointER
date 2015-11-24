package data.nlp.relation;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.Vector;
import java.util.Map.Entry;

import com.google.common.collect.Ordering;

import config.Parameters;

import data.Instance;
import data.Label;
import data.nlp.Document;
import data.nlp.Node;
import data.nlp.Offset;
import data.nlp.Sentence;

public class RelationDocument extends Document {
	protected List<Relation> relations;

	public RelationDocument(Parameters params, String id) {
		this(params, null, null, id, "", "");
	}
	
	public RelationDocument(Parameters params, Document document, Offset offset, String id, String type, String source) {
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
		boolean LtoR = params.useLtoR();
		for(Sentence sentence:sentences){
			//sentence
			if(sentence.getGoldEntities().size() < 2){
				continue;
			}
			Set<Node> entities;
			if(LtoR){
				entities = new TreeSet<Node>(Ordering.natural());
			}else{
				entities = new TreeSet<Node>(Ordering.natural().reverse());
			}
			entities.addAll(sentence.getGoldEntities().values());
			Label label = new Label(params);
			RelationSequence seq = new RelationSequence(params);
			for(Node e1:entities){
				for(Node e2:entities){
					if(!params.getUseSelfRelation() && e1 == e2)continue;
					if(LtoR){
						if(e1.getOffset().compareTo(e2.getOffset()) > 0){
							assert e2.getOffset().compareTo(e1.getOffset()) < 0:e2.getOffset()+":"+e1.getOffset();
							continue;
						}
						if(e1.getOffset().compareTo(e2.getOffset()) == 0){
							if(e1.getId().compareTo(e2.getId()) > 0){
								continue;
							}
						}
					}else{
						if(e1.getOffset().compareTo(e2.getOffset()) < 0){
							assert e2.getOffset().compareTo(e1.getOffset()) > 0:e2.getOffset()+":"+e1.getOffset();
							continue;
						}
						if(e1.getOffset().compareTo(e2.getOffset()) == 0){
							if(e1.getId().compareTo(e2.getId()) < 0){
								continue;
							}
						}
					}
					label.add(getGoldRelation(e1, e2, relArgs));
					seq.add(new EntityPair(params, e1, e2));
				}
			}
			Instance instance = new RelationInstance(params, seq, label, sentence);
			if(params.useCloseFirst() || params.useSort()){
				instance.sort();
			}			
			assert label.size() != 0;
			assert label.size() == seq.size();
			assert label.size() == entities.size() * (entities.size() + 1) / 2 - (params.getUseSelfRelation() ? 0 : entities.size()) : label.size()+":"+entities.size();			
			instances.add(instance);
		}
	}

	private RelationLabelUnit getGoldRelation(Node e1, Node e2, String[] relArgs) {
		Collection<String> rels = e1.getRelations().get(e2);
		EntityPair pair = new EntityPair(params, e1, e2);
		if(rels.size() == 0){
			return RelationLabelUnit.getNegativeClassLabelUnit(params, pair);
		}else{
			return new RelationLabelUnit(params, rels, pair, relArgs);
		}
	}
	
	
	public void addRelation(Relation relation) {
		this.relations.add(relation);	
	}
	
}
