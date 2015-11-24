package model;

import no.uib.cipr.matrix.sparse.SparseVector;

public class VectorInfo{
	private int key;
	private double weight;
	private SparseVector sv;
	public VectorInfo(int key, double weight, SparseVector sv) {
		this.key = key;
		this.weight = weight;
		this.sv = sv;
	}
	public VectorInfo(VectorInfo info) {
		this.key = info.key;
		this.weight = info.weight;
		this.sv = info.sv;
	}
	
	public int getKey() {
		return key;
	}
	public double getWeight() {
		return weight;
	}
	public SparseVector getSv() {
		return sv;
	}
	public void setKey(int key) {
		this.key = key;
	}
	public void scale(double weight) {
		this.weight *= weight;
	}	
}