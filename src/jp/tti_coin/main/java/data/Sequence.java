package data;

public interface Sequence {
	int size();
	SequenceUnit get(int index);
	void add(SequenceUnit unit);
}
