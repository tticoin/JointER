package data.nlp;

public class Offset implements Comparable<Offset>{
	private int start;
	private int end;

	public Offset(int start, int end) {
		assert start <= end;
		assert start >= 0;
		assert end >= 0;
		this.start = start;
		this.end = end;
	}

	public int getStart() {
		return start;
	}

	public int getEnd() {
		return end;
	}

	public int getSize() {
		assert start <= end;
		return end - start;
	}

	@Override
	public int compareTo(Offset rhs) {
		// depends on task ...
		if(getEnd() < rhs.getStart()){
			return -1;
		}else if(getStart() > rhs.getEnd()){
			return 1;
		}else if(includes(rhs)){
			// equals depending on the definition of "includes"
			if(getSize() == rhs.getSize()){
				assert this.getStart() == rhs.getStart() && this.getEnd() == rhs.getEnd();
				return 0;
			}
			assert getSize() > rhs.getSize();
			return -1;
		}else if(rhs.includes(this)){
			assert this.getStart() != rhs.getStart() || this.getEnd() != rhs.getEnd();
			assert getSize() < rhs.getSize();
			return 1;
		}else if(getStart() < rhs.getStart()){
			assert getEnd() < rhs.getEnd();
			return -1;
		}else if(getStart() > rhs.getStart()){
			assert getEnd() > rhs.getEnd();
			return 1;
		}
		assert false;
		return 0;
	}

	public boolean includes(Offset rhs) {
		return start <= rhs.start && rhs.end <= end;
	}

	@Override
	public String toString() {
		return "(" + getStart() + ", " + getEnd() + ")";
	}

	public boolean overlaps(Offset offset) {
		return start <= offset.start && offset.start <= end || offset.start <= start && start <= offset.end;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + end;
		result = prime * result + start;
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Offset other = (Offset) obj;
		if (end != other.end)
			return false;
		if (start != other.start)
			return false;
		return true;
	}

	

}
