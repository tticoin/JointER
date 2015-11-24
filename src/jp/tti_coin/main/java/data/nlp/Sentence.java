package data.nlp;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;
import java.util.Vector;

import com.google.common.collect.Multimap;

import config.Parameters;

public class Sentence extends Nodes {
	private Map<String, Node> goldEntities;
	private Map<String, Map<String, Node>> parseNodes;
	private Map<String, List<Node>> words; 

	public Sentence(Parameters params, Multimap<String, Node> nodes) {
		super(params, nodes);
		this.goldEntities = new TreeMap<String, Node>(); 	
		this.parseNodes = new TreeMap<String, Map<String, Node>>(); 	
		this.words = new TreeMap<String, List<Node>>();
		this.setSentence(this);
		if(isGoldEntity()){
			for(Node goldEntity:this.getNodes(Parameters.getGoldAnnotationDesc())){
				goldEntities.put(goldEntity.getId(), goldEntity);
			}
		}
		for(String parseAnnotationDesc:params.getParseAnnotationDescs()){
			parseNodes.put(parseAnnotationDesc, new TreeMap<String, Node>());
			for(Node node:this.getNodes(parseAnnotationDesc)){
				parseNodes.get(parseAnnotationDesc).put(node.getId(), node);
			}
		}
	}

	public void addNodes(Nodes nodes) {
		super.addNodes(nodes);
		if(nodes.isGoldEntity()){
			for(Node goldEntity:nodes.getNodes(Parameters.getGoldAnnotationDesc())){
				goldEntities.put(goldEntity.getId(), goldEntity);
			}
		}
		for(String parseAnnotationDesc:params.getParseAnnotationDescs()){
			for(Node node:nodes.getNodes(parseAnnotationDesc)){
				parseNodes.get(parseAnnotationDesc).put(node.getId(), node);
			}
		}
	}

	public void processParse(){
		for(String parseAnnotationDesc:params.getParseAnnotationDescs()){
			Map<String, Node> currentParseNodes = parseNodes.get(parseAnnotationDesc);
			words.put(parseAnnotationDesc, new Vector<Node>());
			List<Node> currentWords = words.get(parseAnnotationDesc);
			for(Node parseNode:currentParseNodes.values()){
				String pred = "";
				if(params.getParsePredicateAttributeType(parseAnnotationDesc) != null && 
						parseNode.getAttributes().containsKey(params.getParsePredicateAttributeType(parseAnnotationDesc))){
					pred = parseNode.getAttributes().get(params.getParsePredicateAttributeType(parseAnnotationDesc)).iterator().next();
				}
				if(params.getParseHeadAttributeType(parseAnnotationDesc) == null){
					parseNode.setHead(null);
				}
				for(Entry<String, String> attribute:parseNode.getAttributes().entries()){
					if(currentParseNodes.containsKey(attribute.getValue())){
						assert !attribute.getValue().isEmpty();
						Node argument = currentParseNodes.get(attribute.getValue());
						if(params.getIgnoredAttributeTypes(parseAnnotationDesc) != null && 
								params.getIgnoredAttributeTypes(parseAnnotationDesc).contains(attribute.getKey())){
							continue;
						}else if(attribute.getKey().equals(params.getParseHeadAttributeType(parseAnnotationDesc))){
							parseNode.setHead(argument);
						}else{
							parseNode.addArgument(argument, pred+attribute.getKey());
						}
					}
				}
				if(parseNode.getType().equals(params.getParseWordTag(parseAnnotationDesc))){
					if(!currentWords.contains(parseNode)){
						currentWords.add(parseNode);
					}
				}
			}
			Collections.sort(currentWords);
			for(Node word:currentWords){
				word.calcWordArguments();
			}
			assert currentWords.size() < 2 || currentWords.get(0).getOffset().compareTo(currentWords.get(1).getOffset()) < 0;
		}
	}
	
	public Map<String, Node> getGoldEntities() {
		return goldEntities;
	}

	public List<Node> getWords(String parseAnnotationDesc) {
		return words.get(parseAnnotationDesc);
	}

	public void removeNestedEntities() {
		Map<String, Node> oldGoldEntities = new TreeMap<String, Node>(goldEntities);
		for(Map.Entry<String, Node> e1:oldGoldEntities.entrySet()){
			for(Map.Entry<String, Node> e2:oldGoldEntities.entrySet()){
				if(!e1.getKey().equals(e2.getKey()) && e1.getValue().isInsideOf(e2.getValue())){
					if(params.getVerbosity() > 3){
						System.out.println("Nested entity:" + e1.getKey() + " is ignored!!");
					}
					goldEntities.remove(e1.getKey());
				}
			}			
		}
	}
	
	
}
