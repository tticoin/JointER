package utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.List;
import java.util.Map;
import java.util.NavigableSet;
import java.util.TreeMap;

import org.itadaki.bzip2.BZip2InputStream;
import org.mapdb.Bind;
import org.mapdb.DB;
import org.mapdb.DBMaker;
import org.mapdb.Fun;
import org.mapdb.Fun.Tuple2;

import data.nlp.Node;

public class NICTNounSynonymsDB extends SynonymsDB {
	protected NavigableSet<Tuple2<String, Tuple2<String,Double>>> nounSynonymsDB;

	public NICTNounSynonymsDB(String filename) {
		super(filename);
		nounSynonymsDB = db.getTreeSet("SW");
	}
	
	@Override
	public Map<String, Float> getDirectSynonyms(Node node) {
		Map<String, Float> synonyms = new TreeMap<String, Float>();
		for(Tuple2<String,Double> v:Bind.findVals2(nounSynonymsDB, node.getText())){
			synonyms.put(v.a, v.b.floatValue());
		}
		synonyms.put(node.getText(), 1.f);
		return synonyms;
	}
	
	public Map<String, Float> getSynonyms(Node word){
		Map<String, Float> synonyms = new TreeMap<String, Float>();
		// longest match
		if(word.getPOS().equals("名詞")){
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
				boolean found = false;
				for(Tuple2<String,Double> v:Bind.findVals2(nounSynonymsDB, sb.toString())){
					synonyms.put(v.a, v.b.floatValue());
					found = true;
				}
				if(found){
					synonyms.put(sb.toString(), 1.f);
					break;
				}
			}
			if(current == last){
				synonyms.put(words.get(current).getRealBase(), 1.f);
			}
		}
		return synonyms;
	}

	static void compile(String file, String database){
		DB db = DBMaker.newFileDB(new File(database)).compressionEnable().closeOnJvmShutdown().make();
		NavigableSet<Tuple2<String, Tuple2<String,Double>>> multiset = db.getTreeSet("SW");
		multiset.clear();
		try {
			BufferedReader swFile = new BufferedReader(new InputStreamReader(new BZip2InputStream(new FileInputStream(new File(file)), false), "UTF-8"));
			int entry = 0;
			for(String line=swFile.readLine();line != null;line = swFile.readLine()){
				String key = line.trim();
				String[] values = swFile.readLine().trim().split(" ");
				for(int i = 1;i < values.length;i += 2){
					multiset.add(Fun.t2(key, Fun.t2(values[i-1], Double.parseDouble(values[i]))));
				}
				if(entry++ > 100){
					db.commit();
					entry = 0;
				}
			}
			swFile.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		db.commit();
		db.close();
	}
		
	public static void main(String[] args){
		if(args.length != 2){
			System.err.println("Usage: java "+NICTNounSynonymsDB.class.getName()+ " file db");
			System.exit(-1);
		}
		NICTNounSynonymsDB.compile(args[0], args[1]);
	}

}
