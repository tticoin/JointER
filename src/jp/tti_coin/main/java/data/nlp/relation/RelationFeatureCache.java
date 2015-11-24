package data.nlp.relation;

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

public class RelationFeatureCache extends FeatureCache {
	private static SynonymsDB synonymsDB;
	private final int NGRAM_SIZE = 4; 

	public RelationFeatureCache(Parameters params, Node node) {
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
		for(String parseAnnotationDesc:params.getParseAnnotationDescs()){
			nodeFeatures.add(wordFeatures(parseAnnotationDesc).normalize(), "HEADWORD"+parseAnnotationDesc);			
		}
		
		nodeFeatures.compact();
		nodeFeatures.normalize();
		
		nodeFeatures.add(phraseFeatures().normalize(), "PHRASE");
		assert nodeFeatures.size() > 0;
		nodeFeatures.compact();
		nodeFeatures.normalize();
	}
	
	private StringSparseVector wordAttributeFeatures(Node headWord){
		StringSparseVector features = new StringSparseVector(params);
		for(String tokenFeatureAttributeType:params.getTokenFeatureAttributeTypes(headWord.getSourceDesc())){
			for(String attribute:headWord.getAttributes().get(tokenFeatureAttributeType)){
				features.add(tokenFeatureAttributeType+":"+attribute, 1.);
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
		features.add(wordAttributeFeatures(headWord).normalize(), "WATTR");
		if(params.getUseFullFeatures()){
			features.add(characterNgramFeatures(headWord.getRealBase(), NGRAM_SIZE).normalize(), "CNGRAM");
			if(synonymsDB != null){
				for(Entry<String, Float> synonym:synonymsDB.getSynonyms(headWord).entrySet()){
					features.add("SYN:"+synonym.getKey(), synonym.getValue());
				}
			}
			if(params.getNictVerbEntailmentFile() != null){
				for(String entail:params.getNictVerbEntailments(headWord)){
					features.add("ENTAIL:"+entail, 1.);
				}
			}	
		}
		return features;
	}
	
	private StringSparseVector wordFeatures(String parseAnnotationDesc){
		StringSparseVector features = new StringSparseVector(params);
		Set<Node> words = node.getHeadWords(parseAnnotationDesc);
		for(Node word:words){
			features.add(wordFeatures(word).normalize());
		}
		return features;
	}
	
	private StringSparseVector phraseFeatures(){
		StringSparseVector features = new StringSparseVector(params);
		if(synonymsDB != null){
			for(Entry<String, Float> synonym:synonymsDB.getDirectSynonyms(node).entrySet()){
				features.add("SYN:"+synonym.getKey(), synonym.getValue());
			}
		}
		if(params.getNictVerbEntailmentFile() != null){
			for(String entail:params.getDirectNictVerbEntailments(node)){
				features.add("ENTAIL:"+entail, 1.);
			}
		}			
		return features;
	}

	@Override
	public void calcPathFeatures(Node e2) {
		if(fullPathFeatures != null && fullPathFeatures.containsKey(e2)){
			assert fullPathFeatures.get(e2).size() > 0;
			assert shortestPathFeatures.get(e2).size() > 0;
			return;
		}
		if(fullPathFeatures == null){
			fullPathFeatures = new TreeMap<Node, SparseFeatureVector>();
			shortestPathFeatures = new TreeMap<Node, SparseFeatureVector>();
		}
		SparseFeatureVector fullFeatures = new SparseFeatureVector(params);
		SparseFeatureVector simplePathFeatures = new SparseFeatureVector(params);
		fullFeatures.add(node.getCache().getNodeFeatures(), "E1Node");
		fullFeatures.add(e2.getCache().getNodeFeatures(), "E2Node");
		fullFeatures.compact();
		fullFeatures.normalize();

		for(String parseAnnotationDesc:params.getParseAnnotationDescs()){
			StringSparseVector SPFeatures = shortestPathFeatures(e2, NGRAM_SIZE, parseAnnotationDesc);
			
			fullFeatures.add(ngramPathFeatures(e2, NGRAM_SIZE, NGRAM_SIZE-1, parseAnnotationDesc).normalize(), "PNGRAM"+parseAnnotationDesc);
			fullFeatures.add(SPFeatures, "SPPATH"+parseAnnotationDesc);
			if(params.getUseFullFeatures()){
				simplePathFeatures.add(ngramPathFeatures(e2, NGRAM_SIZE, 0, parseAnnotationDesc).normalize(), "SPNGRAM"+parseAnnotationDesc);
			}
			simplePathFeatures.add(SPFeatures, "SPPATH"+parseAnnotationDesc);
		}
		fullFeatures.compact();
		assert fullFeatures.size() > 0;
		
		fullPathFeatures.put(e2, fullFeatures);
		
		simplePathFeatures.compact();
		simplePathFeatures.normalize();
		assert simplePathFeatures.size() > 0;
		
		shortestPathFeatures.put(e2, simplePathFeatures);
	}
	
	private StringSparseVector ngramPathFeatures(Node e2, int N, int window, String parseAnnotationDesc){
		StringSparseVector features = new StringSparseVector(params);
		List<Node> words = node.getNodes().getSentence().getWords(parseAnnotationDesc);
		int pairStart = Math.min(node.getOffset().getStart(), e2.getOffset().getStart());
		int pairEnd = Math.max(node.getOffset().getEnd(), e2.getOffset().getEnd());
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

		if(params.getUseTermSurface()){
			List<String> surroundings = new Vector<String>();
			assert le < 0 || (!words.get(le).isInsideOf(e2) && !words.get(le).isInsideOf(node)): Math.max(0, le - window + 1)+":"+(le + 1);
			if(le < 0){
				surroundings.add("SS:L");
			}
			for(int i = Math.max(0, le - window + 1);i < le + 1;i++){
				surroundings.add("L:"+getTokenRealBasePOS(words.get(i), e2));
			}
			for(int i = le+1;i < rs;i++){
				surroundings.add("M:"+getTokenRealBasePOS(words.get(i), e2));
			}
			assert rs >= nwords || (!words.get(rs).isInsideOf(e2) && !words.get(rs).isInsideOf(node));
			for(int i = rs;i < Math.min(rs + window, nwords);i++){
				surroundings.add("R:"+getTokenRealBasePOS(words.get(i), e2));
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
			assert le < 0 || (!words.get(le).isInsideOf(e2) && !words.get(le).isInsideOf(node)): Math.max(0, le - window + 1)+":"+(le + 1);
			if(le < 0){
				surroundings.add("SS:L");
			}
			for(int i = Math.max(0, le - window + 1);i < le + 1;i++){
				surroundings.add("L:"+getTokenBasePOS(words.get(i), e2));
			}
			for(int i = le+1;i < rs;i++){
				surroundings.add("M:"+getTokenBasePOS(words.get(i), e2));
			}
			assert rs >= nwords || (!words.get(rs).isInsideOf(e2) && !words.get(rs).isInsideOf(node));
			for(int i = rs;i < Math.min(rs + window, nwords);i++){
				surroundings.add("R:"+getTokenBasePOS(words.get(i), e2));
			}
			if(rs + window > nwords){
				surroundings.add("ES:R");
			}
			int nwindowWords = surroundings.size();
			for(int n = 1;n <= N;n++){
				for(int i = 0;i < nwindowWords - n + 1;i++){
					StringBuffer sb = new StringBuffer();
					sb.append("BASE:");
					for(int j = i;j < i+n;j++){
						sb.append(surroundings.get(j));
						sb.append(":");
					}
					features.add(sb.toString(), 1.);
				}
			}
		}

		if(params.getUseFullFeatures()){
			List<String> surroundings = new Vector<String>();
			if(le < 0){
				surroundings.add("SS:L");
			}
			for(int i = Math.max(0, le - window + 1);i < le + 1;i++){
				surroundings.add("L:"+getTokenPOS(words.get(i), e2));
			}
			for(int i = le+1;i < rs;i++){
				surroundings.add("M:"+getTokenPOS(words.get(i), e2));
			}
			for(int i = rs;i < Math.min(rs + window, nwords);i++){
				surroundings.add("R:"+getTokenPOS(words.get(i), e2));
			}
			if(rs + window > nwords){
				surroundings.add("ES:R");
			}
			int nwindowWords = surroundings.size();
			for(int n = 1;n <= N;n++){
				for(int i = 0;i < nwindowWords - n + 1;i++){
					StringBuffer sb = new StringBuffer();
					sb.append("POS:");
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
	
	private String getTokenRealBasePOS(Node n, Node e2) {
		if(n.isInsideOf(node)){
			return "E1:"+n.getRealBase()+"/"+n.getPOS();
		}
		if(n.isInsideOf(e2)){
			return "E2:"+n.getRealBase()+"/"+n.getPOS();
		}
		return n.getRealBase()+"/"+n.getPOS();
	}
	
	private String getTokenBasePOS(Node n, Node e2) {
		if(n.isInsideOf(node)){
			return "E1:"+n.getBase()+"/"+n.getPOS();
		}
		if(n.isInsideOf(e2)){
			return "E2:"+n.getBase()+"/"+n.getPOS();
		}
		return n.getBase()+"/"+n.getPOS();
	}
	
	private void addEWalks(String e1, String dep, String e2, StringSparseVector features){
		features.add("E:"+e1+":"+dep+":"+e2, 1.);
		features.add("E:"+e1+":"+dep+":", 1.);
		features.add("E:"+e1+"::"+e2, 1.);
		features.add("E::"+dep+":"+e2, 1.);
		features.add("E:"+e1+"::", 1.);
		features.add("E::"+dep+":", 1.);
		features.add("E:::"+e2, 1.);
	}

	private void addVWalks(String dep1, String e, String dep2, StringSparseVector features){
		features.add("V:"+dep1+":"+e+":"+dep2, 1.);
		features.add("V:"+dep1+"::"+dep2, 1.);
	}
	
	private StringSparseVector shortestPathFeatures(Node e2, int N, String parseAnnotationDesc){
		StringSparseVector features = new StringSparseVector(params);
		Dijkstra dijkstra = new Dijkstra(params);
		Set<Node> e1HeadWords = node.getHeadWords(parseAnnotationDesc);
		Set<Node> e2HeadWords = e2.getHeadWords(parseAnnotationDesc);
		assert e1HeadWords.size() > 0 && e2HeadWords.size() > 0: node.getText()+":"+e2.getText();
		int globalShortestPathSize = 100;
		List<List<Node>> globalShortestPaths = null;
		for(Node e1HeadWord:e1HeadWords){
			for(Node e2HeadWord:e2HeadWords){
				if(e1HeadWord.equals(e2HeadWord)){
					globalShortestPathSize = 0;
					globalShortestPaths = null;
					continue;
				}
				List<List<Node>> shortestPaths = dijkstra.getShortestPath(e1HeadWord, e2HeadWord, parseAnnotationDesc);
				if(shortestPaths.size() > 0){
					int localSPSize = shortestPaths.get(0).size();
					if(globalShortestPathSize > localSPSize){
						globalShortestPathSize = localSPSize;
						globalShortestPaths = shortestPaths;
					}
				}
			}
		}

		if(globalShortestPaths != null){
			for(List<Node> shortestPath:globalShortestPaths){		
				assert shortestPath.size() > 1 && shortestPath.size() == globalShortestPathSize;
				Node prev = null;
				String prevDep = null;
				List<String> deps = new Vector<String>();
				List<String> bigramBases = new Vector<String>();
				List<String> bigramRealBases = new Vector<String>();
				List<String> bigramPOSs = new Vector<String>();
				for(int i = 0;i < globalShortestPathSize;i++){
					Node current = shortestPath.get(i);
					if(prev != null){
						String currentDep = prev.getFirstDependency(current); 
						deps.add(currentDep);
						boolean hasDirect = false;	
						if(!currentDep.startsWith(Node.REVERSE_ARGUMENT_HEADER)){
							hasDirect = true;
						}			

						if(params.getUseTermSurface()){
							addEWalks(getTokenRealBase(prev, e2), currentDep, getTokenRealBase(current, e2), features);
							if(prevDep != null){
								addVWalks(prevDep, getTokenRealBase(prev, e2), currentDep, features);
							}
							if(hasDirect){
								bigramRealBases.add(getTokenRealBase(prev, e2)+"->"+getTokenRealBase(current, e2));
							}else{
								bigramRealBases.add(getTokenRealBase(prev, e2)+"<-"+getTokenRealBase(current, e2));
							}
						}
						{
							addEWalks(getTokenBase(prev, e2), currentDep, getTokenBase(current, e2), features);								
							if(prevDep != null){
								addVWalks(prevDep, getTokenBase(prev, e2), currentDep, features);
							}
							if(hasDirect){
								bigramBases.add(getTokenBase(prev, e2)+"->"+getTokenBase(current, e2));
							}else{
								bigramBases.add(getTokenBase(prev, e2)+"<-"+getTokenBase(current, e2));
							}
						}

						if(params.getUseFullFeatures()){
							addEWalks(getTokenPOS(prev, e2), currentDep, getTokenPOS(current, e2), features);	
							if(prevDep != null){
								addVWalks(prevDep, getTokenPOS(prev, e2), currentDep, features);
							}					
							if(hasDirect){
								bigramPOSs.add(getTokenPOS(prev, e2)+"->"+getTokenPOS(current, e2));
							}else{
								bigramPOSs.add(getTokenPOS(prev, e2)+"<-"+getTokenPOS(current, e2));
							}
						}
						prevDep = currentDep;
					}
					prev = current;
				}
				for(int n = 1;n < N;n++){
					if(params.getUseTermSurface()){
						for(int i = 0;i < globalShortestPathSize - n;i++){
							StringBuffer sb = new StringBuffer();
							for(int j = i;j < i+n;j++){
								sb.append(bigramRealBases.get(j));
								sb.append(":");
							}
							features.add("BI:"+sb.toString(), 1.);
						}
					}
					{	
						for(int i = 0;i < globalShortestPathSize - n;i++){
							StringBuffer sb = new StringBuffer();
							for(int j = i;j < i+n;j++){
								sb.append(bigramBases.get(j));
								sb.append(":");
							}
							features.add("BI:"+sb.toString(), 1.);
						}
					}
					if(params.getUseFullFeatures()){
						for(int i = 0;i < globalShortestPathSize - n;i++){
							StringBuffer sb = new StringBuffer();
							for(int j = i;j < i+n;j++){
								sb.append(bigramPOSs.get(j));
								sb.append(":");
							}
							features.add("BI:"+sb.toString(), 1.);
						}
					}
				}
				for(int n = 2;n <= N;n++){
					if(globalShortestPathSize < n){
						continue;
					}	
					if(params.getUseTermSurface()){
						for(int i = 0;i < globalShortestPathSize - n + 1;i++){
							StringBuffer sb = new StringBuffer();
							for(int j = i;j < i+n;j++){
								sb.append(getTokenRealBase(shortestPath.get(j), e2));
								sb.append(":");
							}
							features.add("N:"+sb.toString(), 1.);
						}
					}
					{
						for(int i = 0;i < globalShortestPathSize - n + 1;i++){
							StringBuffer sb = new StringBuffer();
							for(int j = i;j < i+n;j++){
								sb.append(getTokenBase(shortestPath.get(j), e2));
								sb.append(":");
							}
							features.add("N:"+sb.toString(), 1.);
						}		
					}
					if(params.getUseFullFeatures()){
						for(int i = 0;i < globalShortestPathSize - n + 1;i++){
							StringBuffer sb = new StringBuffer();
							for(int j = i;j < i+n;j++){
								sb.append(getTokenPOS(shortestPath.get(j), e2));
								sb.append(":");
							}
							features.add("N:"+sb.toString(), 1.);
						}
					}
					assert globalShortestPathSize == deps.size() + 1;
					for(int i = 0;i < deps.size() - n + 1;i++){
						StringBuffer sb = new StringBuffer();
						for(int j = i;j < i+n;j++){
							sb.append(deps.get(j));
							sb.append(":");
						}
						features.add("N:"+sb.toString(), 3.);
					}
				}
			}
		}else{
			if(globalShortestPathSize == 0){
				features.add("SELF_PATH", 1.);
			}else{
				features.add("NO_PATH", 1.);
			}
		}
		features.normalize();
		features.add("SP"+globalShortestPathSize, 1.);
		return features;
	}

	private String getTokenRealBase(Node n, Node e2) {
		if(n.isInsideOf(node)){
			return "E1:"+n.getRealBase();
		}
		if(n.isInsideOf(e2)){
			return "E2:"+n.getRealBase();
		}
		return n.getRealBase();
	}
	
	private String getTokenBase(Node n, Node e2) {
		if(n.isInsideOf(node)){
			return "E1:"+n.getBase();
		}
		if(n.isInsideOf(e2)){
			return "E2:"+n.getBase();
		}
		return n.getBase();
	}

	private String getTokenPOS(Node n, Node e2) {
		if(n.isInsideOf(node)){
			return "E1:"+n.getPOS();
		}
		if(n.isInsideOf(e2)){
			return "E2:"+n.getPOS();
		}
		return n.getPOS();
	}

	public static void close() {
		if(synonymsDB != null){
			synonymsDB.close();	
			synonymsDB = null;
		}
	}
}
