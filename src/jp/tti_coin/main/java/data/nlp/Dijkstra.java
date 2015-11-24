package data.nlp;

import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;
import java.util.Vector;

import config.Parameters;

public class Dijkstra {
	private Parameters params;
	private Map<Node, DNode> dnodes;
    private class DNode{
        public Map<Node, Float> costs;
        public List<Node> froms;
        public boolean done;
        public Node node;
        public DNode(Node node){
            this.costs = new TreeMap<Node, Float>();
            this.froms = new Vector<Node>();
            this.done = false;
            this.node = node;
        }
        public float cost;
    }
	
    public Dijkstra(Parameters params){
		this.params = params;
    }
    
    public List<List<Node>> getShortestPath(Node n1, Node n2, String parseAnnotationDesc){
    	assert n1 != null;
    	assert n2 != null;
    	assert n1.getNodes().getSentence() == n2.getNodes().getSentence();
    	if(n1 == n2){
            List<List<Node>> allSPs = new Vector<List<Node>>();
            List<Node> sp = new Vector<Node>();
            sp.add(n2);
            allSPs.add(sp);
            return allSPs;
    	}
    	List<Node> words = n1.getNodes().getSentence().getWords(parseAnnotationDesc);
    	assert words.contains(n1) && words.contains(n2);
        dnodes = new TreeMap<Node, DNode>();
        for(Node word:words){
            assert word != null;
            DNode dword = new DNode(word);
            for(Entry<Node, String> toNode:word.getWordArguments().entries()){
            	dword.costs.put(toNode.getKey(), 1.f);
            }
            dword.done = false;
            dword.cost = -1.f;
            dnodes.put(word, dword);
        }
        dnodes.get(n1).cost = 0.f;
        while(true){
            DNode currentDNode = null;
            for(DNode dnode:dnodes.values()){
                if(dnode.done || dnode.cost < 0.f)continue;
                if(currentDNode == null||dnode.cost < currentDNode.cost){
                    currentDNode = dnode;
                }
            }
            if(currentDNode == null)break;
            currentDNode.done = true;
            if(currentDNode.node == n2)break;
            for(Node next:currentDNode.costs.keySet()){
                DNode nextDNode = dnodes.get(next);
                if(nextDNode == null)continue;
                float cost = currentDNode.cost + currentDNode.costs.get(next);
                if(nextDNode.cost < 0 ||cost < nextDNode.cost){
                    nextDNode.froms.clear();
                    nextDNode.froms.add(currentDNode.node);
                    nextDNode.cost = cost;
                }else if(cost == nextDNode.cost){
                    nextDNode.froms.add(currentDNode.node);
                }
            }
        }
        return getAllShortestPaths(n1, n2);
    }		
    
    private List<List<Node>> getAllShortestPaths(Node n1, Node n2) {
        List<List<Node>> allSPs = new Vector<List<Node>>();
        if(n1 == n2){
            List<Node> sp = new Vector<Node>();
            sp.add(n2);
            allSPs.add(sp);
            return allSPs;
        }
        DNode toNode = dnodes.get(n2);
        if(toNode.cost < 0){
            if(params.getVerbosity() > 4){
                System.out.println("Document "+n1.getDocument().getId()+": no path between nodes \""+n1.getText()+":"+n1.getId() + "\" and \""+ n2.getText()+":"+n2.getId()+"\" for "+n1.getSourceDesc());
            }
            return allSPs;
        }
        for(Node node:toNode.froms){
            List<List<Node>> preSPs = getAllShortestPaths(n1, node);
            for(List<Node> preSP:preSPs){
                List<Node> sp = new Vector<Node>(preSP);
                sp.add(n2);
                allSPs.add(sp);
            }
        }
        return allSPs;
    }

}
