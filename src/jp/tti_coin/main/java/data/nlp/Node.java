package data.nlp;

import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import model.FeatureCache;

import com.google.common.collect.Multimap;
import com.google.common.collect.TreeMultimap;

import config.Parameters;

public class Node implements Comparable<Node> {
	public final static String REVERSE_ARGUMENT_HEADER = "<=";
	protected Parameters params;
	protected Document document;
	protected Offset offset;
	protected String id;
	protected String type;
	protected String base;
	protected String pos;
	protected String sourceDesc;
	protected Nodes nodes;
	protected Node headNode;
	protected Map<String, Set<Node>> headWords;
	protected Multimap<String, String> attributes;
	protected Multimap<Node, String> relations;
	protected Multimap<Node, String> arguments;
	protected Multimap<Node, String> wordArguments;
	protected FeatureCache cache;
	public Node(Parameters params, Document document, Offset offset, String id, String type, String sourceDesc){
		this(params, document, offset, id, type, sourceDesc, null);
	}
	public Node(Parameters params, Document document, Offset offset, String id, String type, String sourceDesc, Multimap<String, String> attributes){
		this.params = params;
		this.document = document;
		this.offset = offset;
		this.id = id;
		this.type = type;
		this.sourceDesc = sourceDesc;
		this.headWords = new TreeMap<String, Set<Node>>();
		this.attributes = attributes;
		this.relations = TreeMultimap.create();
		this.arguments = TreeMultimap.create();
		if(params.getParseParameters().containsKey(sourceDesc) && 
				params.getParseBaseAttributeType(sourceDesc) != null &&
				this.attributes.get(params.getParseBaseAttributeType(sourceDesc)).size() > 0){
			this.base = this.attributes.get(params.getParseBaseAttributeType(sourceDesc)).iterator().next();
		}else{
			this.base = "";
		}
		if(params.getParseParameters().containsKey(sourceDesc) && 
				params.getParsePosAttributeType(sourceDesc) != null &&
			this.attributes.get(params.getParsePosAttributeType(sourceDesc)).size() > 0){
			this.pos = this.attributes.get(params.getParsePosAttributeType(sourceDesc)).iterator().next();
		}else{
			this.pos = "";
		}
	}
	public String getId(){
		return id;
	}
	public String getType(){
		return type;
	}
	public String getText(){
		if(!params.getUseTermSurface()){
			if(this.getNodes().isGoldEntity()){
				Node firstEntity = getNodes().getNodes(Parameters.getGoldAnnotationDesc()).iterator().next();
				return firstEntity.getType();
			}else if(this.getNodes().hasGoldParent()){
				Nodes firstParent = this.getNodes().getGoldParents().get(0);
				Node firstEntity = firstParent.getNodes(Parameters.getGoldAnnotationDesc()).iterator().next();
				return firstEntity.getType();
			}else if(this.getNodes().hasGoldChildren()){
				Nodes firstChild = this.getNodes().getGoldChildren().get(0);
				Node firstEntity = firstChild.getNodes(Parameters.getGoldAnnotationDesc()).iterator().next();
				return firstEntity.getType();
			}
		}
		return this.document.getText(offset);
	}	
	public String getRealBase() {
		if(params.getUseTermSurface()){
			if(this.base.isEmpty()){
				base = getText();
			}		
			return base;
		}else{
			return getBase();
		}
	}
	public String getBase() {
		if(this.getNodes().isGoldEntity()){
			Node firstEntity = getNodes().getNodes(Parameters.getGoldAnnotationDesc()).iterator().next();
			return firstEntity.getType();
		}else if(this.getNodes().hasGoldParent()){
			Nodes firstParent = this.getNodes().getGoldParents().get(0);
			Node firstEntity = firstParent.getNodes(Parameters.getGoldAnnotationDesc()).iterator().next();
			return firstEntity.getType();
		}else if(this.getNodes().hasGoldChildren()){
			Nodes firstChild = this.getNodes().getGoldChildren().get(0);
			Node firstEntity = firstChild.getNodes(Parameters.getGoldAnnotationDesc()).iterator().next();
			return firstEntity.getType();
		}else{
			if(this.base.isEmpty()){
				base = getText();
			}		
			return base;
		}
	}
	public String getPOS() {
		return pos;
	}
	public String getSourceDesc() {
		return sourceDesc;
	}
	public FeatureCache getCache() {
		return cache;
	}
	public void setCache(FeatureCache cache) {
		this.cache = cache;
	}
	@Override
	public int compareTo(Node rhs) {
		int comp = offset.compareTo(rhs.offset);
		if(comp != 0)return comp;
		comp = id.compareTo(rhs.id);
		if(comp != 0)return comp;
		comp = type.compareTo(rhs.type);
		if(comp != 0)return comp;
		return sourceDesc.compareTo(rhs.sourceDesc);
	}
	public boolean isInsideOf(Node node) {
		return node.getOffset().includes(this.getOffset());
	}
	public Offset getOffset() {
		return offset;
	}
	public void setNodes(Nodes nodes) {
		this.nodes = nodes;
	}
	public void addGoldRelation(Node node, String rel) {
		relations.put(node, rel);		
		assert relations.get(node).contains(rel);
	}
	public Multimap<String, String> getAttributes() {
		return attributes;
	}
	public void setHead(Node headNode) {
		this.headNode = headNode;
	}
	public void addArgument(Node argument, String key) {
		arguments.put(argument, key);
	}
	public void addWordArgument(Node argument, String key) {
		if(wordArguments == null){
			wordArguments = TreeMultimap.create();	
		}
		wordArguments.put(argument, key);
	}
	public void calcWordArguments(){
		//assert getNodes().getSentence().getWords(this.getSourceDesc()).contains(this);
		if(wordArguments == null){
			wordArguments = TreeMultimap.create();	
		}
		for(Entry<Node, String> argument:arguments.entries()){
			Node headWord = argument.getKey().getHeadWord();
			assert getNodes().getSentence().getWords(this.getSourceDesc()).contains(headWord);
			addWordArgument(headWord, argument.getValue());
			headWord.addWordArgument(this, REVERSE_ARGUMENT_HEADER.concat(argument.getValue()));
		}		
	}
	
	public Nodes getNodes() {
		return nodes;
	}
	
	public Document getDocument(){
		return document;
	}
	
	private Set<Node> getHeadWords(Set<Nodes> targets, String sourceDesc){
		Set<Node> targetHeadWords = getParentHeadWords(targets, sourceDesc);
		if(targetHeadWords.size() != 0){
			return targetHeadWords;
		}
		targetHeadWords = getChildHeadWords(targets, sourceDesc);
		if(targetHeadWords.size() != 0){
			return targetHeadWords;
		}
		//TODO: No directly related words 
		return getOverlappingWords(sourceDesc);
	}
	
	private Set<Node> getOverlappingWords(String sourceDesc){
		// get head from target node;
		Set<Node> headWords = new TreeSet<Node>();
		for(Node word:this.getNodes().getSentence().getWords(sourceDesc)){
			if(word.getOffset().overlaps(this.getOffset())){
				headWords.add(word);
			}
		}
		return headWords;
	}
	
	private Set<Node> getCurrentHeadWords(Set<Nodes> targets, String sourceDesc){
		// get head from target node;
		Set<Node> parseNodes = new TreeSet<Node>();
		List<Node> words = this.getNodes().getSentence().getWords(sourceDesc);
		for(Nodes target:targets){			
			if(target.hasNodes(sourceDesc)){
				parseNodes.addAll(target.getNodes(sourceDesc));
			}
		}
		Set<Node> headWords = new TreeSet<Node>();
		for(Node parseNode:parseNodes){
			for(Node headWord:parseNode.getHeadWords(sourceDesc)){
				if(words.contains(headWord)){
					if(headWord.isInsideOf(this) || this.isInsideOf(headWord)){
						headWords.add(headWord);
					}
				}
			}
		}
		return headWords;
	}
	
	private Set<Node> getParentHeadWords(Set<Nodes> targets, String sourceDesc){
		// get head from parent nodes;
		Set<Node> parentHeadWords = getCurrentHeadWords(targets, sourceDesc);
		if(parentHeadWords.size() != 0){
			return parentHeadWords;
		}
		Set<Nodes> parents = new TreeSet<Nodes>();
		for(Nodes target:targets){			
			parents.addAll(target.getParents());
		}
		if(parents.size() != 0){		
			return getParentHeadWords(parents, sourceDesc);
		}
		return parentHeadWords;		
	}
	
	private Set<Node> getChildHeadWords(Set<Nodes> targets, String sourceDesc){
		// get head from child nodes;
		Set<Node> childHeadWords = getCurrentHeadWords(targets, sourceDesc);
		if(childHeadWords.size() != 0){
			return childHeadWords;
		}
		Set<Nodes> children = new TreeSet<Nodes>();
		for(Nodes target:targets){			
			children.addAll(target.getChildren());
		}
		if(children.size() != 0){		
			return getChildHeadWords(children, sourceDesc);
		}
		return childHeadWords;		
	}
	
	private Node getHeadWord(){
		// get head for parse node;
		if(this.headNode != null){
			return this.headNode.getHeadWord();
		}else{
			return this;
		}
	}
	
	public Node getFirstHeadWord(String sourceDesc){
		return getHeadWords(sourceDesc).iterator().next();
	}
		
	public Set<Node> getHeadWords(String sourceDesc) {
		if(headWords.containsKey(sourceDesc)){
			return headWords.get(sourceDesc);
		}
		Set<Node> heads = new TreeSet<Node>();
		if(this.sourceDesc.equals(sourceDesc)){
			Node headWord = getHeadWord();
			assert headWord.isInsideOf(this): headWord.getText()+":"+this.getText();
			heads.add(headWord);
		}else{
			Set<Nodes> nodes = new TreeSet<Nodes>();
			nodes.add(this.getNodes());
			heads = getHeadWords(nodes, sourceDesc);
		}
		assert heads.size() > 0: sourceDesc+":"+this.getText()+":"+this.sourceDesc;
		headWords.put(sourceDesc, heads);
		return heads;
	}

	public String getFirstDependency(Node node){
		if(!wordArguments.containsKey(node)){
			return null;
		}
		for(String dep:wordArguments.get(node)){
			if(!dep.startsWith(REVERSE_ARGUMENT_HEADER)){
				return dep;
			}
		}
		for(String dep:wordArguments.get(node)){
			return dep;
		}
		return null;
	}
	
	public Multimap<Node, String> getWordArguments() {
		return wordArguments;
	}
	public Multimap<Node, String> getRelations() {
		return relations;
	}

}
