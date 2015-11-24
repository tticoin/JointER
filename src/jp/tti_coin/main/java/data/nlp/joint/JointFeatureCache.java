package data.nlp.joint;

import java.util.List;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeMap;
import java.util.Vector;

import utils.NICTNounSynonymsDB;
import utils.SynonymsDB;

import config.Parameters;

import data.nlp.Dijkstra;
import data.nlp.Node;
import model.FeatureCache;
import model.SparseFeatureVector;
import model.StringSparseVector;

public class JointFeatureCache extends FeatureCache {
	private static SynonymsDB synonymsDB;
	private final int NGRAM_SIZE = 3; 

	public JointFeatureCache(Parameters params, Node node) {
		super(params, node);
		if(params.getNictSynonymFile() != null && synonymsDB == null){
			synonymsDB = new NICTNounSynonymsDB(params.getNictSynonymFile());
		}
	}
	
	@Override
	public void calcNodeFeatures()  {
		if(nodeFeatures != null){
			assert nodeFeatures.size() > 0;
			return;
		}
		nodeFeatures = new SparseFeatureVector(params);
		nodeFeatures.add(wordFeatures(node).normalize(), "WORD");
		for(String parseAnnotationDesc:params.getParseAnnotationDescs()){
			nodeFeatures.add(wordNgramFeatures(node.getFirstHeadWord(parseAnnotationDesc), NGRAM_SIZE, NGRAM_SIZE-1).normalize(1./Math.sqrt(params.getParseAnnotationDescs().size())), "WNGRAM"+parseAnnotationDesc);
		}
		nodeFeatures.compact();
		nodeFeatures.normalize();
		assert nodeFeatures.size() > 0;
	}
	
	private StringSparseVector wordAttributeFeatures(Node headWord){
		StringSparseVector features = new StringSparseVector(params);
		for(String tokenFeatureAttributeType:params.getTokenFeatureAttributeTypes(headWord.getSourceDesc())){
			for(String attribute:headWord.getAttributes().get(tokenFeatureAttributeType)){
				features.add(tokenFeatureAttributeType+":"+attribute, 1.);
			}
		}
		String surface = headWord.getText();
		if(surface.substring(0,1).matches("[A-Z]")){
			features.add("INIT_CAP", 1.);
		}
		if(surface.matches("^.*[A-Z].*$") && surface.toUpperCase().equals(surface)){
			features.add("ALL_CAP", 1.);
		}
		if(surface.matches("\\d+")){
			features.add("ALL_DIGITS", 1.);
		}else if(surface.matches("\\W+")){
			features.add("PUNCT", 1.);
		}else if(surface.matches("[\\d\\W]+")){
			features.add("ALL_DIGITS_PUNCS", 1.);
		}	
		if(!params.getUseByte()){
			if(surface.matches("\\p{InHiragana}+")){
				features.add("HIRAGANA", 1.);
			}else if(surface.matches("\\p{InCjkUnifiedIdeographs}+")){
				features.add("KANJI", 1.);
			}else if(surface.matches("\\p{InKatakana}+")){
				features.add("KATAKANA", 1.);
			}else if(surface.matches("[A-Za-z]")){
				features.add("ENG", 1.);
			}else if(surface.matches("[\\p{InHiragana}\\p{InCjkUnifiedIdeographs}]+")){
				features.add("KANJIHIRAGANA", 1.);
			}
		}
		return features;
	}
	
	private StringSparseVector characterNgramFeatures(String surface, int N){
		StringSparseVector features = new StringSparseVector(params);
		String text = "^"+surface+"$";
		for(int n = 2;n <= N;n++){
			for(int i = 0;i < text.length() - n + 1;i++){
				features.add(text.substring(i, i + n), 1.);
			}
		}
		return features;
	}
	
	private StringSparseVector wordFeatures(Node headWord){
		StringSparseVector features = new StringSparseVector(params);
		for(String parseAnnotationDesc:params.getParseAnnotationDescs()){
			features.add(wordAttributeFeatures(node.getFirstHeadWord(parseAnnotationDesc)), "WATTR"+parseAnnotationDesc);			
		}
		if(params.getUseFullFeatures()){
			features.add(characterNgramFeatures(headWord.getText(), NGRAM_SIZE+1), "CNGRAM");
			StringSparseVector synonymFeatures = new StringSparseVector(params);
			if(synonymsDB != null){
				for(Entry<String, Float> synonym:synonymsDB.getSynonyms(headWord).entrySet()){
					synonymFeatures.add(synonym.getKey(), synonym.getValue());
				}
			}
			features.add(synonymFeatures, "SYN");
			StringSparseVector entailFeatures = new StringSparseVector(params);
			if(params.getNictVerbEntailmentFile() != null){
				for(String entail:params.getNictVerbEntailments(headWord)){
					entailFeatures.add(entail, 1.);
				}
			}
			features.add(entailFeatures, "ENTAIL");
		}
		return features;
	}
		
	private StringSparseVector wordNgramFeatures(Node headWord, int N, int window){
		StringSparseVector features = new StringSparseVector(params);
		assert headWord.getNodes().getSentence() != null: headWord.getId() +":"+headWord.getRealBase();
		List<Node> words = headWord.getNodes().getSentence().getWords(headWord.getSourceDesc());		
		int target = words.indexOf(headWord);
		int le = target - 1;
		int rs = target + 1;
		int nwords = words.size();
		{
			List<String> surroundings = new Vector<String>();
			if(le < 0){
				surroundings.add("SS:L");
			}
			for(int i = Math.max(0, le - window + 1);i < le + 1;i++){
				surroundings.add("L:"+getTokenRealBase(words.get(i)));
			}
			for(int i = le+1;i < rs;i++){
				surroundings.add("M:"+getTokenRealBase(words.get(i)));
			}
			for(int i = rs;i < Math.min(rs + window, nwords);i++){
				surroundings.add("R:"+getTokenRealBase(words.get(i)));
			}
			if(rs + window > nwords){
				surroundings.add("ES:R");
			}
			int nwindowWords = surroundings.size();
			for(int n = 1;n <= N;n++){
				for(int i = 0;i < nwindowWords - n + 1;i++){
					StringBuffer sb = new StringBuffer();
					sb.append("RBASE:");
					for(int j = i;j < i+n;j++){
						sb.append(surroundings.get(j));
						sb.append(":");
					}
					features.add(sb.toString(), 1.);
				}
			}
		}
		{
			List<String> surroundings = new Vector<String>();
			if(le < 0){
				surroundings.add("SS:L");
			}
			for(int i = Math.max(0, le - window + 1);i < le + 1;i++){
				surroundings.add("L:"+getTokenPOS(words.get(i)));
			}
			for(int i = le+1;i < rs;i++){
				surroundings.add("M:"+getTokenPOS(words.get(i)));
			}
			for(int i = rs;i < Math.min(rs + window, nwords);i++){
				surroundings.add("R:"+getTokenPOS(words.get(i)));
			}
			if(rs + window > nwords){
				surroundings.add("ES:R");
			}
			int nwindowWords = surroundings.size();
			for(int n = 1;n <= N;n++){
				for(int i = 0;i < nwindowWords - n + 1;i++){
					StringBuffer sb = new StringBuffer();
					sb.append("RBASE:");
					for(int j = i;j < i+n;j++){
						sb.append(surroundings.get(j));
						sb.append(":");
					}
					features.add(sb.toString(), 1.);
				}
			}
		}
		return features;
	}

	private String getTokenRealBase(Node n) {
		if(n.isInsideOf(node)){
			return "TARGET:"+n.getRealBase();
		}
		return n.getRealBase();
	}
	

	private String getTokenPOS(Node n) {
		if(n.isInsideOf(node)){
			return "TARGET:"+n.getPOS();
		}
		return n.getPOS();
	}
	
	@Override
	public void calcPathFeatures(Node w2) {
		if(fullPathFeatures != null && fullPathFeatures.containsKey(w2)){
			assert fullPathFeatures.get(w2).size() > 0;
			assert shortestPathFeatures.get(w2).size() > 0;
			return;
		}
		if(fullPathFeatures == null){
			fullPathFeatures = new TreeMap<Node, SparseFeatureVector>();
			shortestPathFeatures = new TreeMap<Node, SparseFeatureVector>();
		}
		SparseFeatureVector fullFeatures = new SparseFeatureVector(params);
		SparseFeatureVector simplePathFeatures = new SparseFeatureVector(params);
		fullFeatures.add(wordFeatures(node).normalize(1./Math.sqrt(2.)), "W1Node");
		fullFeatures.add(wordFeatures(w2).normalize(1./Math.sqrt(2.)), "W2Node");
		for(String parseAnnotationDesc:params.getParseAnnotationDescs()){
			StringSparseVector spFeatures = decomposedShortestPathFeatures(w2, NGRAM_SIZE, parseAnnotationDesc).normalize(1./Math.sqrt(params.getParseAnnotationDescs().size()));
			fullFeatures.add(ngramPathFeatures(w2, NGRAM_SIZE, NGRAM_SIZE-1, parseAnnotationDesc).normalize(1./Math.sqrt(params.getParseAnnotationDescs().size())), "PNGRAM"+parseAnnotationDesc);
			fullFeatures.add(spFeatures, "SPPATH"+parseAnnotationDesc);
			simplePathFeatures.add(spFeatures, "SPPATH"+parseAnnotationDesc);
		}
		fullFeatures.compact();
		assert fullFeatures.size() > 0;
		fullFeatures.scale(params.getRelWeight());
		fullFeatures.normalize();
		fullPathFeatures.put(w2, fullFeatures);	
		
		simplePathFeatures.compact();
		simplePathFeatures.normalize(1./Math.sqrt(3));
		shortestPathFeatures.put(w2, simplePathFeatures);		
	}
	
	private StringSparseVector ngramPathFeatures(Node w2, int N, int window, String parseAnnotationDesc){
		StringSparseVector features = new StringSparseVector(params);
		List<Node> words = node.getNodes().getSentence().getWords(parseAnnotationDesc);
		int pairStart = Math.min(node.getOffset().getStart(), w2.getOffset().getStart());
		int pairEnd = Math.max(node.getOffset().getEnd(), w2.getOffset().getEnd());
		int nwords = words.size();
		int le = -1, rs = nwords;
		for(int i = 0;i < nwords;i++){
			if(words.get(i).getOffset().getEnd() <= pairStart){
				le = i;
				continue;
			}
		}
		for(int i = Math.max(0, le);i < nwords;i++){
			if(pairEnd <= words.get(i).getOffset().getStart()){
				rs = i;
				break;
			}
		}
		List<String> surroundings = new Vector<String>();
		assert le < 0 || (!words.get(le).isInsideOf(w2) && !words.get(le).isInsideOf(node)): Math.max(0, le - window + 1)+":"+(le + 1);
		if(le < 0){
			surroundings.add("SS:L");
		}
		for(int i = Math.max(0, le - window + 1);i < le + 1;i++){
			surroundings.add("L:"+getTokenRealBasePOS(words.get(i), w2));
		}
		for(int i = le+1;i < rs;i++){
			surroundings.add("M:"+getTokenRealBasePOS(words.get(i), w2));
		}
		assert rs >= nwords || (!words.get(rs).isInsideOf(w2) && !words.get(rs).isInsideOf(node));
		for(int i = rs;i < Math.min(rs + window, nwords);i++){
			surroundings.add("R:"+getTokenRealBasePOS(words.get(i), w2));
		}
		if(rs + window > nwords){
			surroundings.add("ES:R");
		}
		int nwindowWords = surroundings.size();
		for(int n = 1;n <= N;n++){
			for(int i = 0;i < nwindowWords - n + 1;i++){
				StringBuffer sb = new StringBuffer();
				sb.append("RBASE:");
				for(int j = i;j < i+n;j++){
					sb.append(surroundings.get(j));
					sb.append(":");
				}
				features.add(sb.toString(), 1.);
			}
		}
		return features;
	}
	
	private String getTokenRealBasePOS(Node n, Node w2) {
		if(n.isInsideOf(node)){
			return "W1:"+n.getRealBase()+"/"+n.getPOS();
		}
		if(n.isInsideOf(w2)){
			return "W2:"+n.getRealBase()+"/"+n.getPOS();
		}
		return n.getRealBase()+"/"+n.getPOS();
	}
	
	private void addEWalks(String w1, String dep, String w2, StringSparseVector features){
		features.add("E:"+w1+":"+dep+":"+w2, 1.);
	}

	private void addVWalks(String dep1, String w, String dep2, StringSparseVector features){
		features.add("V:"+dep1+":"+w+":"+dep2, 1.);
	}
	
	private StringSparseVector decomposedShortestPathFeatures(Node w2, int N, String parseAnnotationDesc){
		StringSparseVector features = new StringSparseVector(params);
		Dijkstra dijkstra = new Dijkstra(params);
		Set<Node> w1HeadWords = node.getHeadWords(parseAnnotationDesc);
		Set<Node> w2HeadWords = w2.getHeadWords(parseAnnotationDesc);
		assert w1HeadWords.size() > 0 && w2HeadWords.size() > 0: node.getText()+":"+w2.getText();
		int globalShortestPathSize = 100;
		List<List<Node>> globalShortestPaths = null;
		for(Node w1HeadWord:w1HeadWords){
			for(Node w2HeadWord:w2HeadWords){
				if(w1HeadWord.equals(w2HeadWord)){
					globalShortestPathSize = 0;
					globalShortestPaths = null;
					break;
				}
				List<List<Node>> shortestPaths = dijkstra.getShortestPath(w1HeadWord, w2HeadWord, parseAnnotationDesc);
				if(shortestPaths.size() > 0){
					int localSPSize = shortestPaths.get(0).size();
					if(globalShortestPathSize > localSPSize){
						globalShortestPathSize = localSPSize;
						globalShortestPaths = shortestPaths;
					}
				}
			}
			if(globalShortestPathSize == 0){
				break;
			}
		}

		if(globalShortestPaths != null){
			for(List<Node> shortestPath:globalShortestPaths){		
				assert shortestPath.size() > 1 && shortestPath.size() == globalShortestPathSize;
				Node prev = null;
				String prevDep = null;
				List<String> tokens = new Vector<String>();
				List<String> deps = new Vector<String>();
				List<String> bigramRealBases = new Vector<String>();
				String prevToken = "";
				String currentToken = "";
				for(int i = 0;i < globalShortestPathSize;i++){
					Node current = shortestPath.get(i);
					prevToken = currentToken;
					currentToken = getTokenRealBase(current, w2);
					tokens.add(currentToken);
					if(prev != null){
						String currentDep = prev.getFirstDependency(current); 
						deps.add(currentDep);
						boolean hasDirect = false;	
						if(!currentDep.startsWith(Node.REVERSE_ARGUMENT_HEADER)){
							hasDirect = true;
						}			
						addEWalks(prevToken, currentDep, currentToken, features);
						if(prevDep != null){
							addVWalks(prevDep, prevToken, currentDep, features);
						}
						if(hasDirect){
							bigramRealBases.add(prevToken+"->"+currentToken);
						}else{
							bigramRealBases.add(prevToken+"<-"+currentToken);
						}
						prevDep = currentDep;
					}
					prev = current;
				}
				for(int n = 1;n < N;n++){
					for(int i = 0;i < globalShortestPathSize - n;i++){
						StringBuffer sb = new StringBuffer();
						for(int j = i;j < i+n;j++){
							sb.append(bigramRealBases.get(j));
							sb.append(":");
						}
						features.add("BI:"+sb.toString(), 1.);
					}
				}
				for(int n = 2;n <= N;n++){
					if(globalShortestPathSize < n){
						continue;
					}	
					for(int i = 0;i < tokens.size() - n + 1;i++){
						StringBuffer sb = new StringBuffer();
						for(int j = i;j < i+n;j++){
							sb.append(tokens.get(j));
							sb.append(":");
						}
						features.add("N:"+sb.toString(), 1.);
					}
					assert globalShortestPathSize == deps.size() + 1;
					for(int i = 0;i < deps.size() - n + 1;i++){
						StringBuffer sb = new StringBuffer();
						for(int j = i;j < i+n;j++){
							sb.append(deps.get(j));
							sb.append(":");
						}
						features.add("N:"+sb.toString(), 1.);
					}
				}
			}
			features.add("SP"+globalShortestPathSize, 1.);
		}else{
			if(globalShortestPathSize == 0){
				features.add("SELF_PATH", 1.);
				if(node != w2 && params.getVerbosity() > 3){
					System.out.println("Document "+node.getDocument().getId()+": self path between nodes \""+node.getText()+":"+node.getId() + "\" and \""+ w2.getText()+":"+w2.getId()+"\"");
				}
			}else{
				features.add("NO_PATH", 1.);
			}
		}
		return features;
	}

	private String getTokenRealBase(Node n, Node w2) {
		if(n.isInsideOf(node)){
			return "W1:"+n.getRealBase();
		}
		if(n.isInsideOf(w2)){
			return "W2:"+n.getRealBase();
		}
		return n.getRealBase();
	}
	

	public static void close() {
		if(synonymsDB != null){
			synonymsDB.close();	
			synonymsDB = null;
		}
	}
}
