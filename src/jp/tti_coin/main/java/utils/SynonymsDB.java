package utils;

import java.io.File;
import java.util.Map;

import org.mapdb.DB;
import org.mapdb.DBMaker;

import data.nlp.Node;

public abstract class SynonymsDB {
	protected DB db;
	
	public SynonymsDB(String filename){
		db = DBMaker.newFileDB(new File(filename)).compressionEnable().closeOnJvmShutdown().make();
	}
	
	public abstract Map<String, Float> getSynonyms(Node word);
	public abstract Map<String, Float> getDirectSynonyms(Node node);

	public void close() {
		db.close();		
	}
	
}
