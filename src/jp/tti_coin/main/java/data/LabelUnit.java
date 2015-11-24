package data;

public interface LabelUnit extends Comparable<LabelUnit> {	
	public boolean equals(Object unit);
	public String toString();
	public boolean isNegative();
}
