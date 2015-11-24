package config;

import java.util.List;
import java.util.Vector;

public class ParseParameters {
	static ParseParameters KNP = new ParseParameters();
	static ParseParameters SYNCHA = new ParseParameters();
	static{
		KNP.setSentenceTag("sentence");
		KNP.setSentenceTag("sentence");
		KNP.setExtension(".split.knp.so");
		KNP.setHeadAttributeType("head");
		KNP.setBaseAttributeType("base");
		KNP.setIdAttributeType("id");
		KNP.setPosAttributeType("pos");
		KNP.setPredicateAttributeType("pred");
		KNP.setWordTag("tok");
		List<String> knpAttributeTypes = new Vector<String>();
		List<String> knpIgnoredAttributeTypes = new Vector<String>();
		knpAttributeTypes.add("cat");
		knpAttributeTypes.add("alter");
		knpAttributeTypes.add("category");
		knpAttributeTypes.add("full_category");
		knpAttributeTypes.add("domain");
		KNP.setTokenFeatureAttributeTypes(knpAttributeTypes);
		KNP.setIgnoredAttributeTypes(knpIgnoredAttributeTypes);
	}
	static {
		SYNCHA.setSentenceTag("sentence");
		SYNCHA.setExtension(".split.knp.so");
		SYNCHA.setHeadAttributeType("head");
		SYNCHA.setBaseAttributeType("base");
		SYNCHA.setIdAttributeType("id");
		SYNCHA.setPosAttributeType("pos");
		SYNCHA.setPredicateAttributeType("pred");
		SYNCHA.setWordTag("tok");
		List<String> synchaAttributeTypes = new Vector<String>();
		List<String> synchaIgnoredAttributeTypes = new Vector<String>();
		synchaAttributeTypes.add("cat");
		synchaAttributeTypes.add("NE");
		SYNCHA.setTokenFeatureAttributeTypes(synchaAttributeTypes);
		SYNCHA.setIgnoredAttributeTypes(synchaIgnoredAttributeTypes);
	}	 
	
	
	private String sentenceTag;
	private String extension;
	private String headAttributeType;
	private String baseAttributeType;
	private String idAttributeType;
	private String posAttributeType;
	private String predicateAttributeType;
	private String wordTag;
	private List<String> ignoredAttributeTypes;
	private List<String> tokenFeatureAttributeTypes;
	public String getSentenceTag() {
		return sentenceTag;
	}
	public void setSentenceTag(String sentenceTag) {
		this.sentenceTag = sentenceTag;
	}
	public String getExtension() {
		return extension;
	}
	public void setExtension(String extension) {
		this.extension = extension;
	}
	public String getHeadAttributeType() {
		return headAttributeType;
	}
	public void setHeadAttributeType(String headAttributeType) {
		this.headAttributeType = headAttributeType;
	}
	public String getBaseAttributeType() {
		return baseAttributeType;
	}
	public void setBaseAttributeType(String baseAttributeType) {
		this.baseAttributeType = baseAttributeType;
	}
	public String getIdAttributeType() {
		return idAttributeType;
	}
	public void setIdAttributeType(String idAttributeType) {
		this.idAttributeType = idAttributeType;
	}
	public String getPosAttributeType() {
		return posAttributeType;
	}
	public void setPosAttributeType(String posAttributeType) {
		this.posAttributeType = posAttributeType;
	}
	public String getPredicateAttributeType() {
		return predicateAttributeType;
	}
	public void setPredicateAttributeType(String predicateAttributeType) {
		this.predicateAttributeType = predicateAttributeType;
	}
	public String getWordTag() {
		return wordTag;
	}
	public void setWordTag(String wordTag) {
		this.wordTag = wordTag;
	}
	public List<String> getTokenFeatureAttributeTypes() {
		return tokenFeatureAttributeTypes;
	}
	public void setTokenFeatureAttributeTypes(
			List<String> tokenFeatureAttributeTypes) {
		this.tokenFeatureAttributeTypes = tokenFeatureAttributeTypes;
	}
	public List<String> getIgnoredAttributeTypes() {
		return ignoredAttributeTypes;
	}
	public void setIgnoredAttributeTypes(List<String> ignoredAttributeTypes) {
		this.ignoredAttributeTypes = ignoredAttributeTypes;
	}
	
}
