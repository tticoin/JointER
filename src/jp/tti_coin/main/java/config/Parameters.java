package config;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.Vector;

import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.constructor.Constructor;

import com.google.common.collect.Multimap;
import com.google.common.collect.TreeMultimap;
import com.google.common.io.Files;

import data.LabelUnit;
import data.nlp.Node;
import data.nlp.joint.PairLabelUnit;
import data.nlp.joint.WordLabelUnit;
import data.nlp.relation.RelationLabelUnit;

public class Parameters {
	static class CustomConstructor extends Constructor {
		@Override
		protected List<Object> createDefaultList(int initSize) {
			return new Vector<Object>();
		}
		
		@Override
		protected Map<Object, Object> createDefaultMap() {
			return new TreeMap<Object, Object>();
		}

		@Override
		protected Set<Object> createDefaultSet() {
			return new TreeSet<Object>();
		}
    } 
	public static String getGoldAnnotationDesc() {
		return GOLD_ANNOTATION_DESC;
	}
	public static String getLearningMethods(int i) {
		assert 0 <= i && i < LEARNING_METHODS.length;
		return LEARNING_METHODS[i];
	}
	public static String getNegativeClassLabel(){
		return NEGATIVE_CLASS_LABEL;
	}
	public static String getSearchMethods(int i) {
		assert 0 <= i && i < SEARCH_METHODS.length;
		return SEARCH_METHODS[i];
	}
	
	public static Parameters load(String yamlFile) {
		try {
			Parameters params = (Parameters) new Yaml(new CustomConstructor()).loadAs(Files.newReader(new File(yamlFile), Charset.forName("UTF-8")), Parameters.class);
			assert params != null;
			params.check();
			if(params.getVerbosity() > 2){
				System.out.print(LEARNING_METHODS[params.getLearningMethod()]);
				System.out.print(", "+UPDATE_METHODS[params.getUpdateMethod()]);
				System.out.print(", "+SEARCH_METHODS[params.getSearchMethod()]);
				if(params.useCloseFirst() || params.useSort()){
					System.out.print(" ("+params.getCloseAnnotation()+")");
				}
				
				System.out.println(" options are selected.");
			}
			return params;
		} catch (FileNotFoundException e) {
			System.err.println("Cannot find " + yamlFile);
			e.printStackTrace();
			System.exit(0);
		} catch (Exception e){
			System.err.println("Cannot interpret " + yamlFile);
			e.printStackTrace();
			System.exit(0);
		}
		assert false;
		return null;
	}
	private static final String GOLD_ANNOTATION_DESC = "GOLD";
	private static final String NEGATIVE_CLASS_LABEL = "NEGATIVE";
	private static final String[] LEARNING_METHODS = {"Perceptron", "SGDSVM", "AdaGrad", "AROW", "SCW", "DCDSSVM"};

	private static final String[] SEARCH_METHODS = {"LtoR", "RtoL", "CloseFirst", "EasyFirst", "DifficultFirst", "RandomFirst", "EntityLtoR", "EntityRtoL", "CloseFirst(RtoL)", "EasyEntityFirst", "EasyCloseFirst"};
	private static final String[] UPDATE_METHODS = {"MaxViolation", "Early", "Standard"};
	private int fvSize;
	private int fvBitSize;
	private int verbosity;
	private int beamSize;
	private int iteration;
	private int localIteration;
	private int outputInterval;	
	private int miniBatch;
	private int processors;
	
	private boolean useByte;
	private boolean useSelfRelation;
	private boolean useBestFirst;
	private boolean useParallel;
	private boolean useFullFeatures;
	private boolean useSimpleFeatures;
	private boolean useTermSurface;
	private boolean useDirectedRelation;
	private boolean useEntityTypedLabel;
	private boolean useEntityType;
	private boolean useDynamicSort;
	private boolean useGoldEntitySpan;
	private boolean useRelationTypeFilter;
	
	private boolean useLocalInit;
	private boolean useWeighting;
	private boolean useAveraging;
	private boolean useGlobalFeatures;
	private boolean useGlobalEntityFeatures;
	private boolean useGlobalRelationFeatures;
	private boolean useWeightedMargin;
	
	private double margin;
	private double averageNumWords;
	private double relWeight;
	private double globalWeight;
	private double epsilon;
	
	private double lambda;
	private int learningMethod;
	private int searchMethod;
	private int updateMethod;
	private String trainFolder;
	private String devFolder;
	private String testFolder;
	private String trainVectorFile;
	private String devVectorFile;
	
	private String modelFile;
	private String nictSynonymFile;
	private String nictVerbEntailmentFile;
	private String logBilinearFile;
	private String textExtension;
	private String annotationExtension;
	private String predAnnotationExtension;
	private String sentenceAnnotation;
	private String closeAnnotation;
	
	private Map<String, ParseParameters> parseParameters;
	
	private Multimap<String, LabelUnit> possibleLabels;
	
	private Multimap<String, String> nictVerbEntailmentMap;

	public Parameters(){
		//initialize
		setFvBitSize(20);
		setVerbosity(3);
		setBeamSize(10);
		setIteration(50);
		setOutputInterval(1);
		setMiniBatch(5);
		setProcessors(1);
		setUseByte(false);
		setUseSelfRelation(false);
		setLearningMethod(1);
		setUpdateMethod(0);
		setUseAveraging(true);
		setUseBestFirst(false);
		setUseDirectedRelation(true);
		setUseEntityType(true);
		setUseEntityTypedLabel(true);
		setUseGlobalFeatures(true);
		setUseGlobalEntityFeatures(true);
		setUseGlobalRelationFeatures(true);
		setUseGoldEntitySpan(false);
		setUseWeightedMargin(false);
		setUseFullFeatures(true);
		setUseSimpleFeatures(false);
		setUseTermSurface(true);
		setUseParallel(false);
		setUseLocalInit(false);
		setUseRelationTypeFilter(true);
		setLocalIteration(5);
		setUseWeighting(false);
		setUseDynamicSort(false);
		setMargin(1.0);
		setLambda(1.0e-4);
		setEpsilon(0.);
		setTrainFolder("train/");
		setDevFolder("dev/");
		setTestFolder("test/");
		setModelFile("model.txt");
		setTrainVectorFile(null);
		setDevVectorFile(null);
		setNictSynonymFile(null);
		setNictVerbEntailmentFile(null);
		setLogBilinearFile(null);
		setTextExtension(".split.txt");
		setAnnotationExtension(".split.ann");
		setPredAnnotationExtension(".split.pred.ann");
		setSearchMethod(2);
		setSentenceAnnotation("KNP");
		setCloseAnnotation(null);
		Map<String, ParseParameters> parseParameters = new TreeMap<String, ParseParameters>();
		parseParameters.put("KNP", ParseParameters.KNP);
		parseParameters.put("SYNCHA", ParseParameters.SYNCHA);
		setParseParameters(parseParameters);
		setAverageNumWords(0.);
		setRelWeight(1.);
		setGlobalWeight(1.);
	}
	
	public void check(){
		//TODO: check parameters
		if(this.getUseDynamicSort() && !this.useSort()){
			System.err.println("Dynamic sort is not available for the search method.");
			System.exit(-1);
		}
		if(this.getUseDynamicSort() && !this.getUseGlobalFeatures()){
			System.err.println("Dynamic sort is not available for the local features.");
			System.exit(-1);
		}
		if(this.useSGDSVM()){
			if(this.getLambda() == 1.){
				System.err.println("lambda should not be 1 for SGDSVM.");
				System.exit(-1);
			}
		}
	}
	
	public int fvSize() {
		return fvSize;
	}
	public String getAnnotationExtension() {
		return annotationExtension;
	}
	public double getAverageNumWords() {
		return this.averageNumWords;
	}
	
	public int getBeamSize() {
		return beamSize;
	}
	public String getCloseAnnotation() {
		return closeAnnotation;
	}
	public String getDevFolder() {
		return devFolder;
	}
	public String getDevVectorFile() {
		return devVectorFile;
	}

	public Set<String> getDirectNictVerbEntailments(Node node) {
		Set<String> entailments = new TreeSet<String>();
		String phrase = node.getText().replaceAll("する$", "");
		if(nictVerbEntailmentMap.containsKey(phrase)){
			entailments.add(phrase);
			for(String entailment:nictVerbEntailmentMap.get(phrase)){
				entailments.add(entailment);
			}
		}else{
			entailments.add(phrase);
		}
		return entailments;
	}
	public double getEpsilon() {
		return epsilon;
	}

	public int getFvBitSize() {
		return fvBitSize;
	}
	public double getGlobalWeight() {
		return globalWeight;
	}
	public List<String> getIgnoredAttributeTypes(String sourceDesc) {
		assert parseParameters.containsKey(sourceDesc);
		return parseParameters.get(sourceDesc).getIgnoredAttributeTypes();
	}
	public int getIteration() {
		return iteration;
	}
	
	public double getLambda() {
		return lambda;
	}
	private int getLearningMethod(){
		return learningMethod;
	}
	public int getLocalIteration() {
		return localIteration;
	}
	public double getMargin() {
		return margin;
	}
	public int getMiniBatch() {
		return miniBatch;
	}
	public String getModelFile() {
		return modelFile;
	}
	public String getNictSynonymFile() {
		return nictSynonymFile;
	}
	public String getNictVerbEntailmentFile() {
		return nictVerbEntailmentFile;
	}
	public Set<String> getNictVerbEntailments(Node word) {
		Set<String> entailments = new TreeSet<String>();
    	// longest match
		if(word.getPOS().equals("動詞")|| word.getPOS().equals("名詞")){
			List<Node> words = word.getNodes().getSentence().getWords(word.getSourceDesc());
			int last = words.indexOf(word);
			int current = last;
			while(current > 0){
				current--;
				String ppos = words.get(current).getPOS();
				if(ppos.equals("助詞")){
					current++;
					break;
    			}
    		}   
			for(;current <= last;current++){
				StringBuffer sb = new StringBuffer();
				for(int i = current;i <= last;i++){
					if(words.get(i).getRealBase().matches(" +"))continue;
					sb.append(words.get(i).getRealBase());
    			}
				String phrase = sb.toString().replaceAll("する$", "");
				if(nictVerbEntailmentMap.containsKey(phrase)){
					entailments.add(phrase);
					for(String entailment:nictVerbEntailmentMap.get(phrase)){
						entailments.add(entailment);
					}
			    	break;
    			}
    		}
			if(current == last){
				entailments.add(words.get(current).getRealBase());
			}
    	}    
		return entailments;		
	}
	public int getOutputInterval() {
		return outputInterval;
	}
	public Set<String> getParseAnnotationDescs() {
		return parseParameters.keySet();
	}
	public String getParseBaseAttributeType(String sourceDesc) {
		assert parseParameters.containsKey(sourceDesc);
		return parseParameters.get(sourceDesc).getBaseAttributeType();
	}
	public String getParseExtension(String sourceDesc) {
		assert parseParameters.containsKey(sourceDesc);
		return parseParameters.get(sourceDesc).getExtension();
	}
	public String getParseHeadAttributeType(String sourceDesc) {
		assert parseParameters.containsKey(sourceDesc);
		return parseParameters.get(sourceDesc).getHeadAttributeType();
	}
	public String getParseIdAttributeType(String sourceDesc) {
		assert parseParameters.containsKey(sourceDesc);
		return parseParameters.get(sourceDesc).getIdAttributeType();
	}
	public Map<String, ParseParameters> getParseParameters() {
		return parseParameters;
	}
	public String getParsePosAttributeType(String sourceDesc) {
		assert parseParameters.containsKey(sourceDesc);
		return parseParameters.get(sourceDesc).getPosAttributeType();
	}
	public String getParsePredicateAttributeType(String sourceDesc) {
		assert parseParameters.containsKey(sourceDesc);
		return parseParameters.get(sourceDesc).getPredicateAttributeType();
	}
	
	public String getParseSentenceTag(String sourceDesc) {
		assert parseParameters.containsKey(sourceDesc);
		return parseParameters.get(sourceDesc).getSentenceTag();
	}

	public String getParseWordTag(String sourceDesc) {
		return parseParameters.get(sourceDesc).getWordTag();
	}

	public Multimap<String, LabelUnit> getPossibleLabels() {
		return possibleLabels;
	}

	public Collection<LabelUnit> getPossibleLabels(String unitType) {
		return possibleLabels.get(unitType);
	}
	
	public String getPredAnnotationExtension() {
		return predAnnotationExtension;
	}

	public int getProcessors() {
		return processors;
	}

	public double getRelWeight(){
		return relWeight;
	}

	public int getSearchMethod() {
		return searchMethod;
	}

	public String getSentenceAnnotation() {
		return sentenceAnnotation;
	}

	public String getTestFolder() {
		return testFolder;
	}

	public String getTextExtension() {
		return textExtension;
	}
	
	public List<String> getTokenFeatureAttributeTypes(String sourceDesc) {
		assert parseParameters.containsKey(sourceDesc);
		return parseParameters.get(sourceDesc).getTokenFeatureAttributeTypes();
	}
	
	public String getTrainFolder() {
		return trainFolder;
	}
	
	public String getTrainVectorFile() {
		return trainVectorFile;
	}

	public int getUpdateMethod() {
		return updateMethod;
	}

	public boolean getUseAveraging() {
		return useAveraging;
	}

	public boolean getUseBestFirst() {
		return useBestFirst;
	}

	public boolean getUseByte() {
		return useByte;
	}	
	
	public boolean getUseDirectedRelation() {
		return useDirectedRelation;
	}
	
	public boolean getUseDynamicSort(){
		return useDynamicSort;
	}

	public boolean getUseEntityType() {
		return useEntityType;
	}
	public boolean getUseEntityTypedLabel() {
		return useEntityTypedLabel;
	}

	public boolean getUseFullFeatures() {
		return useFullFeatures;
	}

	public boolean getUseGlobalEntityFeatures() {
		return useGlobalEntityFeatures;
	}	
	
	public boolean getUseGlobalFeatures() {
		return useGlobalFeatures;
	}
	
	public boolean getUseGlobalRelationFeatures() {
		return useGlobalRelationFeatures;
	}

	public boolean getUseGoldEntitySpan(){
		return useGoldEntitySpan;
	}	

	public boolean getUseLocalInit() {
		return useLocalInit;
	}
	
	public boolean getUseParallel() {
		return useParallel;
	}

	public boolean getUseSelfRelation() {
		return useSelfRelation;
	}
	
	public boolean getUseSimpleFeatures() {
		return useSimpleFeatures;
	}
	
	public boolean getUseTermSurface() {
		return useTermSurface;
	}

	public boolean getUseWeightedMargin() {
		return useWeightedMargin;
	}
		
	public boolean getUseWeighting() {
		return useWeighting;
	}
	
	public int getVerbosity() {
		return verbosity;
	}

	public void loadModelParameters(BufferedReader reader) {
		try{
			this.setAverageNumWords(Double.valueOf(reader.readLine().trim()));
			possibleLabels = TreeMultimap.create();
			String line = reader.readLine().trim();
			String[] entries = line.split("\t");
			String type = entries[0];
			if(type.equals("RelationLabelUnit")){
				for(int i = 1;i < entries.length;i+=2){
					String[] labelUnit = entries[i+1].split("\\$");
					assert labelUnit.length > 1: entries[i+1];
					possibleLabels.put(entries[i], new RelationLabelUnit(labelUnit[0], Boolean.valueOf(labelUnit[1])));
				}
			}else if(type.equals("JointLabelUnit")){
				for(int i = 1;i < entries.length;i+=3){
					if(entries[i+1].equals("W")){
						possibleLabels.put(entries[i], new WordLabelUnit(entries[i+2]));
					}else if(entries[i+1].equals("P")){
						possibleLabels.put(entries[i], new PairLabelUnit(entries[i+2]));
					}else{
						assert false;
					}
				}
			}else{
				assert false;
			}
		
		}catch(IOException ex){
			ex.printStackTrace();
		}
	}	
	

	public boolean getUseRelationTypeFilter() {
		return useRelationTypeFilter;
	}
	public void setUseRelationTypeFilter(boolean useRelationTypeFilter) {
		this.useRelationTypeFilter = useRelationTypeFilter;
	}
	
	public void saveModelParameters(BufferedWriter writer) {
		try {
			LabelUnit firstLabelUnit = possibleLabels.values().iterator().next();
			writer.write(String.valueOf(this.getAverageNumWords()));
			writer.write("\n");
			if(firstLabelUnit instanceof RelationLabelUnit){
				writer.write(possibleLabels.values().iterator().next().getClass().getSimpleName());
				writer.write("\t");
				for(Entry<String, LabelUnit> possibleLabel:possibleLabels.entries()){
					writer.write(possibleLabel.getKey());
					writer.write("\t");
					writer.write(possibleLabel.getValue().toString());
					writer.write("\t");
				}
				writer.write("\n");
			}else if(firstLabelUnit instanceof WordLabelUnit || firstLabelUnit instanceof PairLabelUnit){
				writer.write("JointLabelUnit");
				writer.write("\t");
				for(Entry<String, LabelUnit> possibleLabel:possibleLabels.entries()){
					writer.write(possibleLabel.getKey());
					writer.write("\t");
					if(possibleLabel.getValue() instanceof WordLabelUnit){
						writer.write("W\t");
						writer.write(((WordLabelUnit)possibleLabel.getValue()).getLabel());
					}else if(possibleLabel.getValue() instanceof PairLabelUnit){
						writer.write("P\t");
						writer.write(((PairLabelUnit)possibleLabel.getValue()).getLabel());
					}else{
						assert false;
					}
					writer.write("\t");
				}
				writer.write("\n");					
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}	
	
	public void saveProperties(String propertyFile){
		try {
			new Yaml().dump(this, Files.newWriter(new File(propertyFile), Charset.forName("UTF-8")));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	public void setAnnotationExtension(String annotationExtension) {
		this.annotationExtension = annotationExtension;
	}

//	public Collection<String> getEntityTypes() {
//		return entityTypes;
//	}
//	public void setEntityTypes(Collection<String> entityTypes) {
//		this.entityTypes = entityTypes;
//	}
//	
	public void setAverageNumWords(double averageNumWords) {
		this.averageNumWords = averageNumWords;		
	}
	

	public void setBeamSize(int beamSize) {
		this.beamSize = beamSize;
	}

	public void setCloseAnnotation(String closeAnnotation) {
		this.closeAnnotation = closeAnnotation;
	}


	
	public void setDevFolder(String devFolder) {
		this.devFolder = devFolder;
	}

	public void setDevVectorFile(String devVectorFile) {
		this.devVectorFile = devVectorFile;
	}

	public void setEpsilon(double epsilon) {
		assert epsilon >= 0. && epsilon <= 1.;
		this.epsilon = epsilon;
	}
	
	public void setFvBitSize(int fvBitSize) {
		this.fvBitSize = fvBitSize;
		this.fvSize = 1 << fvBitSize;
		assert fvSize > 0 && (fvSize & (fvSize - 1)) == 0;
	}
	public void setGlobalWeight(double globalWeight) {
		this.globalWeight = globalWeight;
	}
	public void setIteration(int iteration) {
		assert iteration >= 0;
		this.iteration = iteration;
	}

	public void setLambda(double lambda) {
		assert lambda >= 0.;
		this.lambda = lambda;
	}

	public void setLearningMethod(int learningMethod) {
		assert learningMethod < LEARNING_METHODS.length;
		this.learningMethod = learningMethod;
	}
	
	public void setLocalIteration(int localIteration) {
		this.localIteration = localIteration;
	}
	public void setMargin(double margin) {
		assert margin >= 0.;
		this.margin = margin;
	}
	
	public void setMiniBatch(int miniBatch) {
		this.miniBatch = miniBatch;
	}

	public void setModelFile(String modelFile) {
		this.modelFile = modelFile;
	}

	public void setNictSynonymFile(String nictSynonymFile) {
		assert nictSynonymFile == null || (new File(nictSynonymFile)).exists();
		this.nictSynonymFile = nictSynonymFile;
	}
	
	public void setNictVerbEntailmentFile(String nictVerbEntailmentFile) {
		this.nictVerbEntailmentFile = nictVerbEntailmentFile;
		if(this.nictVerbEntailmentFile != null){
			this.nictVerbEntailmentMap = TreeMultimap.create();
			try {
				BufferedReader reader = Files.newReader(new File(this.nictVerbEntailmentFile), Charset.forName("UTF-8"));
				for(String line = reader.readLine();line != null;line = reader.readLine()){
					String[] entailmentRules = line.trim().split(" ");
					if(entailmentRules.length != 2){
						continue;
					}
	                if(entailmentRules[0].startsWith("N")){
	                	entailmentRules[0] = entailmentRules[0].substring(1);
	                }
	                if(entailmentRules[1].startsWith("N")){
	                	entailmentRules[1] = entailmentRules[1].substring(1);
	                }      
	                entailmentRules[0] = entailmentRules[0].replaceAll("する$", "");
	                entailmentRules[1] = entailmentRules[1].replaceAll("する$", "");
	                if(entailmentRules[0].isEmpty() || entailmentRules[1].isEmpty())continue;
	                if(entailmentRules[0].equals(entailmentRules[1]))continue;
	                nictVerbEntailmentMap.put(entailmentRules[0], entailmentRules[1]);
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

	public void setOutputInterval(int outputInterval) {
		this.outputInterval = outputInterval;
	}
	

	public void setParseParameters(Map<String, ParseParameters> parseParameters) {
		this.parseParameters = parseParameters;
	}

	public void setPossibleLabels(Multimap<String, LabelUnit> possibleLabels) {
		this.possibleLabels = possibleLabels;		
	}

	public void setPredAnnotationExtension(String predAnnotationExtension) {
		this.predAnnotationExtension = predAnnotationExtension;
	}

	public void setProcessors(int processors) {
		this.processors = processors;
	}

	public void setRelWeight(double relWeight){
		this.relWeight = relWeight;
	}
	
	public void setSearchMethod(int searchMethod){
		assert 0 <= searchMethod && searchMethod < SEARCH_METHODS.length;
		this.searchMethod = searchMethod;
	}
	
	public void setSentenceAnnotation(String sentenceAnnotation) {
		this.sentenceAnnotation = sentenceAnnotation;
	}
	
	public void setTestFolder(String testFolder) {
		this.testFolder = testFolder;
	}
	
	public void setTextExtension(String textExtension) {
		this.textExtension = textExtension;
	}
	public void setTrainFolder(String trainFolder) {
		this.trainFolder = trainFolder;
	}
	public void setTrainVectorFile(String trainVectorFile) {
		this.trainVectorFile = trainVectorFile;
	}
	public void setUpdateMethod(int updateMethod) {
		assert 0 <= updateMethod && updateMethod < UPDATE_METHODS.length;
		this.updateMethod = updateMethod;
	}
	public void setUseAveraging(boolean useAveraging) {
		this.useAveraging = useAveraging;
	}
	public void setUseBestFirst(boolean useBestFirst){
		this.useBestFirst = useBestFirst;
	}
	public void setUseByte(boolean useByte) {
		this.useByte = useByte;
	}
	public void setUseDirectedRelation(boolean useDirectedRelation) {
		this.useDirectedRelation = useDirectedRelation;
	}	
	public void setUseDynamicSort(boolean useDynamicSort){
		this.useDynamicSort = useDynamicSort;
	}
	public void setUseEntityType(boolean useEntityType) {
		this.useEntityType = useEntityType;
	}
	public void setUseEntityTypedLabel(boolean useEntityTypedLabel) {
		this.useEntityTypedLabel = useEntityTypedLabel;
	}
	public void setUseFullFeatures(boolean useFullFeatures) {
		this.useFullFeatures = useFullFeatures;
	}
	public void setUseGlobalEntityFeatures(boolean useGlobalEntityFeatures) {
		this.useGlobalEntityFeatures = useGlobalEntityFeatures;
	}
	public void setUseGlobalFeatures(boolean useGlobalFeatures) {
		this.useGlobalFeatures = useGlobalFeatures;
	}
	public void setUseGlobalRelationFeatures(boolean useGlobalRelationFeatures) {
		this.useGlobalRelationFeatures = useGlobalRelationFeatures;
	}
	public void setUseGoldEntitySpan(boolean useGoldEntitySpan){
		this.useGoldEntitySpan = useGoldEntitySpan;
	}
	public void setUseLocalInit(boolean useLocalInit) {
		this.useLocalInit = useLocalInit;
	}
	public void setUseParallel(boolean useParallel) {
		this.useParallel = useParallel;
	}
	public void setUseSelfRelation(boolean useSelfRelation) {
		this.useSelfRelation = useSelfRelation;
	}
	public void setUseSimpleFeatures(boolean useSimpleFeatures) {
		this.useSimpleFeatures = useSimpleFeatures;
	}
	public void setUseTermSurface(boolean useTermSurface) {
		this.useTermSurface = useTermSurface;
	}	
	public void setUseWeightedMargin(boolean useWeightedMargin) {
		this.useWeightedMargin = useWeightedMargin;
	}
	public void setUseWeighting(boolean useWeighting) {
		this.useWeighting = useWeighting;
	}
	public void setVerbosity(int verbosity) {
		this.verbosity = verbosity;
	}
	public boolean useAdaGrad(){
		return getLearningMethod() == 2;
	}
	public boolean useAROW(){
		return getLearningMethod() == 3;
	}
	public boolean useCloseFirst() {
		return getSearchMethod() == 2;
	}
	public boolean useCloseFirstRtoL() {
		return getSearchMethod() == 8;
	}
	public boolean useDCDSSVM(){
		return getLearningMethod() == 5;
	}
	public boolean useDifficultFirst() {
		return getSearchMethod() == 4;
	}
	public boolean useEarlyUpdate() {
		return getUpdateMethod() == 1;
	}
	public boolean useEasyCloseFirst() {
		return getSearchMethod() == 10;
	}
	public boolean useEasyEntityFirst() {
		return getSearchMethod() == 9;
	}	
	public boolean useEasyFirst() {
		return getSearchMethod() == 3;
	}
	public boolean useEntityLtoR() {
		return getSearchMethod() == 6;
	}
	public boolean useEntityRtoL() {
		return getSearchMethod() == 7;
	}
	public boolean useLtoR() {
		return getSearchMethod() == 0;
	}
	public boolean useMaxViolationUpdate() {
		return getUpdateMethod() == 0;
	}
	public boolean usePerceptron(){
		return getLearningMethod() == 0;
	}
	public boolean useRandomFirst() {
		return getSearchMethod() == 5;
	}	
	public boolean useRtoL() {
		return getSearchMethod() == 1;
	}
	public boolean useSCW(){
		return getLearningMethod() == 4;
	}
	public boolean useSGDSVM(){
		return getLearningMethod() == 1;
	}
	public boolean useSort(){
		return useEasyFirst() || useDifficultFirst() || useRandomFirst() || useEasyEntityFirst() || useEasyCloseFirst();
	}
	public boolean useStandardUpdate() {
		return getUpdateMethod() == 2;
	}

	public String getLogBilinearFile() {
		return logBilinearFile;
	}
	public void setLogBilinearFile(String logBilinearFile) {
		this.logBilinearFile = logBilinearFile;
	}
}
