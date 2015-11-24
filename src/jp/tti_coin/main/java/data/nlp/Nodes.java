package data.nlp;

import java.util.Collection;
import java.util.List;
import java.util.Vector;

import com.google.common.collect.Multimap;

import config.Parameters;

public class Nodes implements Comparable<Nodes> {
	Parameters params;
	Multimap<String, Node> nodes;
	private Node node = null;
	private Sentence sentence = null;
	List<Nodes> parents;
	List<Nodes> children;
	boolean hasGoldParent;
	List<Nodes> goldParents;
	boolean hasGoldChildren;
	List<Nodes> goldChildren;
	
	public Nodes(Parameters params, Multimap<String, Node> nodes) {
		this.params = params;
		this.nodes = nodes;
		for(Node node:nodes.values()){
			node.setNodes(this);
			if(this.getNode() == null){
				this.node = node;
			}
		}
		assert nodes.size() > 0;
		assert this.getNode() != null;
		this.parents = new Vector<Nodes>();
		this.children = new Vector<Nodes>();
		this.goldParents = new Vector<Nodes>();
		this.hasGoldParent = false;
	}

	public Offset getOffset() {
		return getNode().getOffset();
	}
	
	@Override
	public int compareTo(Nodes rnode) {
		return getOffset().compareTo(rnode.getOffset());
	}

	public void addNodes(Nodes nodes) {
		assert this != nodes;
		if(isGoldEntity()){
			nodes.addGoldParents(this);
			nodes.setHasGoldParent(true);
		}
		boolean directParent = true; 
		for(Nodes child:children){
			if(child == nodes){
				// different path due to graph structure.
				return;
			}
			if(child.getOffset().includes(nodes.getOffset())){
				directParent = false;
				child.addNodes(nodes);
			}
		}
		if(directParent){
			children.add(nodes);
			nodes.setParent(this);
		}
	}

	private void addGoldParents(Nodes nodes) {
		goldParents.add(0, nodes);
	}

	public List<Nodes> getGoldParents() {
		return goldParents;
	}

	private void setHasGoldParent(boolean hasGoldParent) {
		this.hasGoldParent = hasGoldParent;
	}

	private void setParent(Nodes parent) {
		parents.add(parent);
	}

	public Node getNode() {
		return node;
	}

	public boolean isGoldEntity() {
		if(this.nodes.containsKey(Parameters.getGoldAnnotationDesc())){
			return true;
		}
		return false;
	}

	public boolean hasNodes(String desc){
		return this.nodes.containsKey(desc);
	}
	
	public Collection<Node> getNodes(String desc){
		return this.nodes.get(desc);
	}

	public void setSentence(Sentence sentence) {
		this.sentence = sentence;
	}

	public Sentence getSentence() {
		return sentence;
	}

	public List<Nodes> getParents() {
		return parents;
	}
	
	public List<Nodes> getChildren() {
		return children;
	}

	public boolean hasGoldParent() {
		return hasGoldParent;
	}
	
	public boolean hasGoldChildren(){
		if(this.goldChildren !=  null){
			return hasGoldChildren;
		}
		this.goldChildren = new Vector<Nodes>();
		if(this.isGoldEntity()){
			hasGoldChildren = true;
			this.goldChildren.add(this);
			return true;
		}
		if(children.size() == 0){
			hasGoldChildren = false;
			return false;
		}
		for(Nodes nodes:children){
			if(nodes.hasGoldChildren()){
				goldChildren.addAll(nodes.getGoldChildren());
				hasGoldChildren = true;
				return true;
			}
		}
		hasGoldChildren = false;
		return false;
	}
	
	public List<Nodes> getGoldChildren() {
		return goldChildren;
	}
	
}
