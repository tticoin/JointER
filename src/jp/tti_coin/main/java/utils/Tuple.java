package utils;

public class Tuple<T1 extends Comparable<T1>, T2 extends Comparable<T2>> implements Comparable<Tuple<T1, T2>> {
	private T1 t1;
	private T2 t2;
	public Tuple(T1 t1, T2 t2) {
		this.t1 = t1;
		this.t2 = t2;
	}
	public T1 getT1() {
		return t1;
	}
	public T2 getT2() {
		return t2;
	}
	@Override
	public int compareTo(Tuple<T1, T2> o) {
		int comp = t1.compareTo(o.t1);
		if(comp != 0)return comp;
		return t2.compareTo(o.t2);
	}
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((t1 == null) ? 0 : t1.hashCode());
		result = prime * result + ((t2 == null) ? 0 : t2.hashCode());
		return result;
	}
	
}
