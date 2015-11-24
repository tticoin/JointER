package data.nlp.relation;

import com.google.common.collect.Multimap;

import config.Parameters;

import data.nlp.Document;

public class Relation {
	private Parameters params;
	private Document document;
	private String id;
	private String type;
	private Multimap<String, String> arguments;
	private String sourceDesc;
	public Relation(Parameters params, Document document, String id, String type, Multimap<String, String> arguments, String sourceDesc) {
		this.params = params;
		this.document = document;
		this.id = id;
		this.type = type;
		this.arguments = arguments;
		this.sourceDesc = sourceDesc;
	}
	public String getSourceDesc() {
		return sourceDesc;
	}
	public String getType() {
		return type;
	}
	public Parameters getParams() {
		return params;
	}
	public Document getDocument() {
		return document;
	}
	public String getId() {
		return id;
	}
	public Multimap<String, String> getArguments() {
		return arguments;
	}
	
}
