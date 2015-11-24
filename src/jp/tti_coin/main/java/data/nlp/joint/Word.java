package data.nlp.joint;

import java.util.Collection;
import java.util.Iterator;
import java.util.Set;
import java.util.TreeSet;

import config.Parameters;
import data.Label;
import data.LabelUnit;
import data.SequenceUnit;
import data.nlp.Node;

public class Word extends SequenceUnit {
	private int id;
	private Node word;
	public static final String TYPE = "WORDTYPE";
	
	public Word(Parameters params, Node word, int id){
		this.params = params;
		this.word = word;
		this.id = id;
	}
	
	@Override
	public String getType() {
		return TYPE;
	}

	@Override
	public LabelUnit getNegativeClassLabel() {
		return WordLabelUnit.getNegativeClassLabelUnit();
	}

	public int getId() {
		return id;
	}
	
	public Collection<LabelUnit> getPossibleLabels() {
		assert false;
		return null;
	}
	
	public Collection<LabelUnit> getRelWordLabels(Label label, JointInstance instance){
		//TODO: no relations??
		Set<LabelUnit> units = new TreeSet<LabelUnit>();
		boolean hasRel = false;
		for(int wi = 0;wi < instance.getNumWords();++wi){
			int pairIdx;
			if(wi < id){
				pairIdx = instance.getWordPair(wi, id);
			}else if(wi > id){
				pairIdx = instance.getWordPair(id, wi);
			}else{
				assert wi == id;
				if(params.getUseSelfRelation()){
					pairIdx = instance.getWordPair(wi, id);
				}else{
					continue;
				}
			}
			if(pairIdx < label.size()){
				assert label.getLabel(pairIdx) instanceof PairLabelUnit;
				if(!label.getLabel(pairIdx).isNegative()){
					PairLabelUnit pairLabelUnit = (PairLabelUnit)label.getLabel(pairIdx);
					units.addAll(params.getPossibleLabels().get(pairLabelUnit.toString()+":"+ (wi < id ? "2" : "1")));
					hasRel = true;
				}
			}
		}		
		assert hasRel || units.size() == 0;
		return units;
	}
	
	public Collection<LabelUnit> getPossibleLabels(Label label, JointInstance instance) {
		int size = label.size();
		Collection<LabelUnit> labels = getRelWordLabels(label, instance);
		boolean hasRelation = labels.size() > 0;
		if(!hasRelation){
			labels = new TreeSet<LabelUnit>(params.getPossibleLabels().get(getType()));
		}
		if(params.getUseGoldEntitySpan()){
			String goldPosition = ((WordLabelUnit)instance.getGoldLabel().getLabel(instance.getWord(id))).getPosition();
			for(Iterator<LabelUnit> unitIt=labels.iterator();unitIt.hasNext();){
				WordLabelUnit unit = (WordLabelUnit)unitIt.next();
				if(!unit.getPosition().equals(goldPosition)){
					unitIt.remove();
				}
			}
			String goldType = instance.getGoldType(label, id);
			if(!goldType.equals("")){
				for(Iterator<LabelUnit> unitIt=labels.iterator();unitIt.hasNext();){
					WordLabelUnit unit = (WordLabelUnit)unitIt.next();
					if(!unit.getType().equals(goldType)){
						unitIt.remove();
					}
				}
			}
		}
		if(id > 0){
			int prevWordIndex = instance.getWord(id - 1);
			Collection<LabelUnit> relLabelUnits = ((Word)instance.getSequence().get(prevWordIndex)).getRelWordLabels(label, instance);
			if(relLabelUnits.size() != 0){
				for(Iterator<LabelUnit> unitIt=labels.iterator();unitIt.hasNext();){
					WordLabelUnit unit = (WordLabelUnit)unitIt.next();
					if(unit.getPosition().equals(WordLabelUnit.I) || 
						unit.getPosition().equals(WordLabelUnit.L)){
						unitIt.remove();
					}
				}
				// B-*, O, U-*					
			}
			if(prevWordIndex < size){				
				assert label.getLabel(prevWordIndex) instanceof WordLabelUnit;
				String position = ((WordLabelUnit)label.getLabel(prevWordIndex)).getPosition();
				String type = ((WordLabelUnit)label.getLabel(prevWordIndex)).getType();
				if(position.equals(WordLabelUnit.B) || position.equals(WordLabelUnit.I)){
					for(Iterator<LabelUnit> unitIt=labels.iterator();unitIt.hasNext();){
						WordLabelUnit unit = (WordLabelUnit)unitIt.next();
						if(!unit.getType().equals(type) || 
								unit.getPosition().equals(WordLabelUnit.B) || 
								unit.getPosition().equals(WordLabelUnit.U) || 
								unit.getPosition().equals(WordLabelUnit.O) ){
							unitIt.remove();
						}
					}
					// I-type, L-type
				}else{
					for(Iterator<LabelUnit> unitIt=labels.iterator();unitIt.hasNext();){
						WordLabelUnit unit = (WordLabelUnit)unitIt.next();
						if(unit.getPosition().equals(WordLabelUnit.I) || 
								unit.getPosition().equals(WordLabelUnit.L)){
							unitIt.remove();
						}
					}
					// B-*, O, U-*
				}
			}
			if(id > 1){
				int pprevWordIndex = instance.getWord(id - 2);
				if(pprevWordIndex < size){
					String position = ((WordLabelUnit)label.getLabel(pprevWordIndex)).getPosition();
					String type = ((WordLabelUnit)label.getLabel(pprevWordIndex)).getType();
					if(position.equals(WordLabelUnit.B)||position.equals(WordLabelUnit.I)){
						for(Iterator<LabelUnit> unitIt=labels.iterator();unitIt.hasNext();){
							WordLabelUnit unit = (WordLabelUnit)unitIt.next();
							if((unit.getPosition().equals(WordLabelUnit.L)||unit.getPosition().equals(WordLabelUnit.I)) && 
									!unit.getType().equals(type)){
								unitIt.remove();
							}
						}
						//B-*, O, I-type, U-*, L-type
					}
				}
			}
		}else if(id == 0){
			for(Iterator<LabelUnit> unitIt=labels.iterator();unitIt.hasNext();){
				WordLabelUnit unit = (WordLabelUnit)unitIt.next();
				if(unit.getPosition().equals(WordLabelUnit.I) || 
						unit.getPosition().equals(WordLabelUnit.L)){
					unitIt.remove();
				}
			}
			// B-*, O, U-*
		}
		if(id < instance.getNumWords() - 1){
			int nextWordIndex = instance.getWord(id + 1);
			Collection<LabelUnit> relLabelUnits = ((Word)instance.getSequence().get(nextWordIndex)).getRelWordLabels(label, instance);
			if(relLabelUnits.size() != 0){
				Set<String> possibleTypes = new TreeSet<String>();
				for(LabelUnit relLabelUnit:relLabelUnits){
					possibleTypes.add(((WordLabelUnit)relLabelUnit).getType());				
				}
				for(Iterator<LabelUnit> unitIt=labels.iterator();unitIt.hasNext();){
					WordLabelUnit unit = (WordLabelUnit)unitIt.next();
					if((unit.getPosition().equals(WordLabelUnit.B) || 
						unit.getPosition().equals(WordLabelUnit.I)) && 
						!possibleTypes.contains(unit.getType())){
						unitIt.remove();
					}
				}
				// B-TYPE, I-TYPE, L, O, U
			}
			if(nextWordIndex < size){
				assert label.getLabel(nextWordIndex) instanceof WordLabelUnit;
				String position = ((WordLabelUnit)label.getLabel(nextWordIndex)).getPosition();
				String type = ((WordLabelUnit)label.getLabel(nextWordIndex)).getType();
				if(position.equals(WordLabelUnit.I) || position.equals(WordLabelUnit.L)){
					for(Iterator<LabelUnit> unitIt=labels.iterator();unitIt.hasNext();){
						WordLabelUnit unit = (WordLabelUnit)unitIt.next();
						if(!unit.getType().equals(type) || 
								unit.getPosition().equals(WordLabelUnit.L) || 
								unit.getPosition().equals(WordLabelUnit.U) || 
								unit.getPosition().equals(WordLabelUnit.O) ){
							unitIt.remove();
						}
					}
					// B-type, I-type
				}else{
					for(Iterator<LabelUnit> unitIt=labels.iterator();unitIt.hasNext();){
						WordLabelUnit unit = (WordLabelUnit)unitIt.next();
						if(unit.getPosition().equals(WordLabelUnit.B) || 
								unit.getPosition().equals(WordLabelUnit.I)){
							unitIt.remove();
						}
					}
					// L-*, O, U-*
				}
			}
			if(id < instance.getNumWords() - 2){
				int nNextWordIndex = instance.getWord(id + 2);
				if(nNextWordIndex < size){
					String position = ((WordLabelUnit)label.getLabel(nNextWordIndex)).getPosition();
					String type = ((WordLabelUnit)label.getLabel(nNextWordIndex)).getType();
					if(position.equals(WordLabelUnit.L) || position.equals(WordLabelUnit.I)){
						//B-type, O, I-type, U-*, L-*
						for(Iterator<LabelUnit> unitIt=labels.iterator();unitIt.hasNext();){
							WordLabelUnit unit = (WordLabelUnit)unitIt.next();
							if((unit.getPosition().equals(WordLabelUnit.B)||unit.getPosition().equals(WordLabelUnit.I)) && 
									!unit.getType().equals(type)){
								unitIt.remove();
							}
						}					
					}
				}
			}
		}else if(id == instance.getNumWords() - 1){
			// L-*, O, U-*
			for(Iterator<LabelUnit> unitIt=labels.iterator();unitIt.hasNext();){
				WordLabelUnit unit = (WordLabelUnit)unitIt.next();
				if(unit.getPosition().equals(WordLabelUnit.B) || 
						unit.getPosition().equals(WordLabelUnit.I)){
					unitIt.remove();
				}
			}
		}
		if(labels.size() == 0 && params.getVerbosity() > 0){
			System.out.println("rel? "+hasRelation+" , id: "+id);
			System.out.println(getRelWordLabels(label, instance));
			for(int i = 0;i < instance.getNumWords();i++){
				int index = instance.getWord(i);
				if(index >= size){
					System.out.print(" "+i+":"+index+":?");
				}else{
					System.out.print(" "+i+":"+index+":"+((WordLabelUnit)label.getLabel(index)).getLabel());
				}
			}
			System.out.println();
		}		
		assert labels.size() > 0: id;
		return labels;
	}

	public Node getWord() {
		return word;
	}
}
