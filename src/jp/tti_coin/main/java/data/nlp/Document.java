package data.nlp;

import java.io.UnsupportedEncodingException;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.Vector;

import com.google.common.collect.Multimap;
import com.google.common.collect.TreeMultimap;

import config.Parameters;

import data.Instance;

public class Document extends Node {
	protected byte[] byteText;
	protected String utfText;
	protected List<Sentence> sentences;
	protected Multimap<Offset, Node> nodes;
	protected List<Instance> instances;
	
	public Document(Parameters params, String id) {
		this(params, null, null, id, "", "");
	}
	
	public Document(Parameters params, Document document, Offset offset, String id, String type, String source) {
		super(params, document, offset, id, type, source);
		assert document == null && offset == null;
		this.sentences = new Vector<Sentence>();
		this.nodes = TreeMultimap.create();
		this.instances = new Vector<Instance>();
	}
	
	public void setText(byte[] text){
		if(params.getUseByte()){
			byteText = text;
			this.offset = new Offset(0, text.length);
		}else{
			try {
				utfText = new String(text, "UTF-8");
				this.offset = new Offset(0, utfText.length());
			} catch (UnsupportedEncodingException e) {
				if(params.getVerbosity() > 0){
					e.printStackTrace();
				}
				System.exit(-1);
			}
		}
	}
	
	public String getText(){
		return getText(this.offset);
	}
	
	public String getText(Offset offset) {
		if(params.getUseByte()){
			return new String(byteText, offset.getStart(), offset.getSize());
		}else{
			return utfText.substring(offset.getStart(), offset.getEnd());
		}
	}

	public void addNode(Node node) {
		nodes.put(node.getOffset(), node);
	}


	public void buildTree() {
		Map<Offset, Nodes> children = new TreeMap<Offset, Nodes>();
		// merge same offset nodes
		for(Offset offset:nodes.keySet()){
			Collection<Node> sharedNodes = nodes.get(offset);
			Multimap<String, Node> nodeMap = TreeMultimap.create();
			for(Node node:sharedNodes){
				nodeMap.put(node.getSourceDesc(), node);
			}
			boolean sentence = false;
			if(nodeMap.containsKey(params.getSentenceAnnotation())){
				for(Node node:nodeMap.get(params.getSentenceAnnotation())){
					if(node.getType().equals(params.getParseSentenceTag(params.getSentenceAnnotation()))){
						sentence = true;
					}
				}
			}
			if(sentence){
				Sentence s = new Sentence(params, nodeMap);
				sentences.add(s);
			}else{
				Nodes nodes = new Nodes(params, nodeMap);
				assert !children.containsKey(offset): offset.getStart()+":"+offset.getEnd()+":"+nodes.getNode().getId()+":"+children.get(offset).getNode().getId();
				children.put(offset, nodes);
			}
		}
		assert children.size() + sentences.size() == nodes.keySet().size();
		assert sentences.size() > 0: "no sentences in documents";
		Collections.sort(sentences);
		// add sentence nodes
		int sentenceId = 0;
		Sentence currentSentence = sentences.get(sentenceId);
		for(Nodes nodes:children.values()){
			if(currentSentence.getOffset().includes(nodes.getOffset())){
				nodes.setSentence(currentSentence);
				currentSentence.addNodes(nodes);
			}else{
				while(currentSentence.getOffset().getEnd() <= nodes.getOffset().getStart() && sentenceId < sentences.size() - 1){
					sentenceId++;
					currentSentence = sentences.get(sentenceId);
				}
				if(currentSentence.getOffset().includes(nodes.getOffset())){
					nodes.setSentence(currentSentence);
					currentSentence.addNodes(nodes);
				}else{
					if(params.getVerbosity() > 3){	
						System.err.println("Document " + id+": a node " + nodes.getNode().id + " from " + nodes.getNode().sourceDesc + " is ignored.");	
					}
				}
			}
		}
		for(Sentence sentence:sentences){
			sentence.removeNestedEntities();
		}
	}

	public List<Instance> getInstances() {
		return instances;
	}

	
}
