package data.nlp.joint;

import java.io.BufferedReader;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.Vector;

import com.google.common.collect.Multimap;
import com.google.common.collect.TreeMultimap;
import com.google.common.io.Files;

import config.Parameters;
import data.Instance;
import data.Label;
import data.LabelUnit;
import data.Sequence;
import data.nlp.Document;
import data.nlp.NLPData;
import data.nlp.Node;
import data.nlp.Offset;
import data.nlp.relation.Relation;

public class JointData extends NLPData {

	public JointData(Parameters params, String baseDir, boolean isTrain) {
		super(params, baseDir, isTrain);
	}
	
	protected JointDocument createDocument(Parameters params, String fileBase) {
		return new JointDocument(params, fileBase);
	}
	
	@Override
	protected void load(String baseDir) {
		documents = new Vector<Document>();
		File directory = new File(baseDir);
		assert directory.isDirectory();
		File[] files = directory.listFiles(new FilenameFilter(){
				public boolean accept(File dir, String name) {
					File file = new File(dir.getAbsolutePath()+File.separator+name); 
					return file.isFile() && name.endsWith(params.getTextExtension());
				}
			});

		for(File file:files){
			String fileBase = file.getAbsolutePath().substring(0, file.getAbsolutePath().length()-params.getTextExtension().length());
			JointDocument document = createDocument(params, fileBase);
			// read text file
			try {
				byte[] texts = Files.toByteArray(new File(fileBase+params.getTextExtension()));
				document.setText(texts);
			} catch (IOException e) {
				if(params.getVerbosity() > 0){
					e.printStackTrace();
				}
			}
			for(String sourceDesc:params.getParseAnnotationDescs()){
				try{
					BufferedReader parserReader = null;
					if(params.getUseByte()){
						parserReader = Files.newReader(new File(fileBase+params.getParseExtension(sourceDesc)), Charset.forName("US-ASCII"));
					}else{
						parserReader = Files.newReader(new File(fileBase+params.getParseExtension(sourceDesc)), Charset.forName("UTF-8"));
					}
					for(String line = parserReader.readLine();line != null;line = parserReader.readLine()){
						String[] entry = line.trim().split("\t");
						if(entry.length < 3)continue;//start end tag 
						Offset offset = new Offset(Integer.parseInt(entry[0]), Integer.parseInt(entry[1]));
						String type;
						String attribute;
						if(entry.length < 4){
							String[] annotation = entry[2].split(" ", 2);
							type = annotation[0];
							attribute = annotation[1];
						}else{
							type = entry[2];
							attribute = entry[3];							
						}
						Multimap<String, String> attributes = TreeMultimap.create();
						String id = "";
						int attStart = 0;
						for(int index = 0;index < attribute.length();index++){
							while(attribute.charAt(index) != '='){
								index++;
							}
							String key = attribute.substring(attStart, index).trim();
							index+=2;
							int valueStart = index;
							while(attribute.charAt(index) != '"' && index < attribute.length()){
								if(attribute.charAt(index) == '\\' && attribute.charAt(index+1) == '"'){
									index += 2;
								}else{
									index++;
								}									
							}
							assert valueStart <= index : valueStart+":"+index;
							String value = attribute.substring(valueStart, index);
							attStart = index+1;
							if(key.equals(params.getParseIdAttributeType(sourceDesc))){
								id = value;
							}else{
								attributes.put(key, value);
							}
						}
						assert !id.equals("");
						Node node = new Node(params, document, offset, id, type, sourceDesc, attributes);
						document.addNode(node);
					}
					parserReader.close();
				}  catch (IOException e) {
					if(params.getVerbosity() > 0){
						e.printStackTrace();
					}
				}
			}
			// read annotation (only R and T annotations)
			try {
				BufferedReader annotationReader = null;
				File annotationFile = new File(fileBase+params.getAnnotationExtension());
				if(annotationFile.exists()){
					if(params.getUseByte()){
						annotationReader = Files.newReader(annotationFile, Charset.forName("US-ASCII"));
					}else{
						annotationReader = Files.newReader(annotationFile, Charset.forName("UTF-8"));
					}
					for(String line = annotationReader.readLine();line != null;line = annotationReader.readLine()){
						if(line.startsWith("T")){
							String[] entry = line.trim().split("\t");
							assert entry.length >= 2: line;
							String id = entry[0];
							String[] annotation = entry[1].split(" ");
							assert annotation.length == 3: line;
							String type = annotation[0];
							if(!params.getUseEntityType()){
								type = "TERM";
							}
							Offset offset = new Offset(Integer.parseInt(annotation[1]), Integer.parseInt(annotation[2]));
							Node entity = new Node(params, document, offset, id, type, Parameters.getGoldAnnotationDesc());
							document.addNode(entity);
						}else if(line.startsWith("R")){
							String[] entry = line.trim().split("\t");
							assert entry.length >= 2: line;
							String id = entry[0];
							String[] annotation = entry[1].split(" ");
							String type = annotation[0];
							Multimap<String, String> arguments = TreeMultimap.create();
							for(int i = 1;i < annotation.length;i++){
								String[] argument = annotation[i].split(":");
								assert argument.length == 2:argument;
								arguments.put(argument[0], argument[1]);
							}
							Relation relation = new Relation(params, document, id, type, arguments, Parameters.getGoldAnnotationDesc());
							document.addRelation(relation);
						}else if(line.startsWith("#")){
							if(params.getVerbosity() > 3){
								System.err.println("Ignore : "+line);
							}
						}else{
							assert false : "Unsupported annotation: "+line;
						}
					}				
					annotationReader.close();
				}else if(params.getVerbosity() > 2){
					System.err.println("Annotations for "+fileBase +" were not found.");
				}
			} catch (IOException e) {
				if(params.getVerbosity() > 0){
					e.printStackTrace();
				}
			}
			document.buildTree();
			document.buildRelations(isTrain);
			documents.add(document);
			instances.addAll(document.getInstances());
		}
		int instanceIndex = 0;
		for(Instance instance:instances){
			instance.setIndex(instanceIndex++);
			instance.setData(this);
		}
		// obtain relation candidates from data 
		if(isTrain){
			Multimap<String, LabelUnit> possibleLabels = TreeMultimap.create();
			double averageNumWords = 0.;
			int nInstances = 0;
			for(Document document:documents){
				for(Instance instance:document.getInstances()){
					averageNumWords += ((JointInstance)instance).getNumWords();
					nInstances++;
					Label label = instance.getGoldLabel();
					Sequence sequence = instance.getSequence();
					int size = label.size();
					for(int i = 0;i < size;i++){
						if(sequence.get(i) instanceof Word){
							possibleLabels.put(sequence.get(i).getType(), label.getLabel(i));
						}else if(sequence.get(i) instanceof Pair){
							JointInstance jInstance = (JointInstance)instance;
							Pair pair = (Pair)sequence.get(i);
							WordLabelUnit w1LabelUnit = (WordLabelUnit)label.getLabel(jInstance.getWord(pair.getW1().getId()));
							WordLabelUnit w2LabelUnit = (WordLabelUnit)label.getLabel(jInstance.getWord(pair.getW2().getId()));
							String w1type = w1LabelUnit.getType();
							String w2type = w2LabelUnit.getType();
							possibleLabels.put(":", label.getLabel(i));
							possibleLabels.put(w1type+":", label.getLabel(i));
							possibleLabels.put(":"+w2type, label.getLabel(i));
							possibleLabels.put(w1type+":"+w2type, label.getLabel(i));
							possibleLabels.put(label.getLabel(i).toString()+":1", new WordLabelUnit(WordLabelUnit.U, w1LabelUnit.getType()));
							possibleLabels.put(label.getLabel(i).toString()+":1", new WordLabelUnit(WordLabelUnit.L, w1LabelUnit.getType()));
							possibleLabels.put(label.getLabel(i).toString()+":2", new WordLabelUnit(WordLabelUnit.U, w2LabelUnit.getType()));
							possibleLabels.put(label.getLabel(i).toString()+":2", new WordLabelUnit(WordLabelUnit.L, w2LabelUnit.getType()));
						}
					}
				}
			}
			averageNumWords /= nInstances;
			params.setAverageNumWords(averageNumWords);
			Set<String> types = new TreeSet<String>();
			for(LabelUnit labelUnit:possibleLabels.get(Word.TYPE)){
				assert labelUnit instanceof WordLabelUnit;
				WordLabelUnit wordLabelUnit = (WordLabelUnit)labelUnit;
				if(wordLabelUnit.isNegative())continue;
				types.add(wordLabelUnit.getType());
			}
			for(String type:types){
				possibleLabels.put(Word.TYPE, new WordLabelUnit(WordLabelUnit.B, type));
				possibleLabels.put(Word.TYPE, new WordLabelUnit(WordLabelUnit.I, type));
				possibleLabels.put(Word.TYPE, new WordLabelUnit(WordLabelUnit.L, type));
				possibleLabels.put(Word.TYPE, new WordLabelUnit(WordLabelUnit.U, type));
			}
			params.setPossibleLabels(possibleLabels);
			if(params.getUseWeightedMargin()||params.getUseWeighting()){
				this.labelImportance = new TreeMap<LabelUnit, Double>();
				for(int i = 0;i < size();i++){
					Instance instance = getInstance(i);
					for(int j = 0;j < instance.size();j++){
						LabelUnit goldLabelUnit = instance.getGoldLabel().getLabel(j);
						if(labelImportance.containsKey(goldLabelUnit)){
							labelImportance.put(goldLabelUnit, 1.+labelImportance.get(goldLabelUnit));
						}else{
							labelImportance.put(goldLabelUnit, 1.);
						}
					}
				}
				double wordCount = 0., pairCount = 0;
				int wordTypes = 0, pairTypes = 0;
				for(Entry<LabelUnit, Double> entry:labelImportance.entrySet()){
					if(entry.getKey() instanceof WordLabelUnit){
						wordCount += entry.getValue();
						wordTypes++;
					}else if(entry.getKey() instanceof PairLabelUnit){
						pairCount += entry.getValue();
						pairTypes++;
					}else{
						assert false;
					}
				}
				for(Entry<LabelUnit, Double> entry:labelImportance.entrySet()){
					//entry.setValue(val/entry.getValue());
					double p = 0.;
					if(entry.getKey() instanceof WordLabelUnit){
						p = (wordCount - entry.getValue()) / entry.getValue();
						p /= wordTypes;
					}else if(entry.getKey() instanceof PairLabelUnit){
						p = (pairCount - entry.getValue()) / entry.getValue();
						p /= pairTypes;
					}
					entry.setValue(p);
				}
			}
		}
	}
}
