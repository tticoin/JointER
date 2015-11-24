package model;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.Arrays;

import config.Parameters;

public class WeightVector{
	private Parameters params;
	private double[] values;
	public WeightVector(Parameters params){
		this.params = params;
		int dim = this.params.fvSize();
		values = new double[dim];
		Arrays.fill(values, 0.);
	}

	public WeightVector(WeightVector weight){
		values = new double[weight.values.length];
		Arrays.fill(values, 0.);
		add(weight);
	}
	
	public void add(int key, double w) {
		assert key >= 0 && key < values.length;
		values[key] += w;
	}
	

	public void set(int key, double w) {
		assert key >= 0 && key < values.length;
		values[key] = w;
	}
	
	public void save(BufferedWriter writer) {
		try {
			writer.write(String.valueOf(values.length));
			for(int i = 0;i < values.length;i++){
				if(Math.abs(values[i]) > Double.MIN_VALUE){
					writer.write(" ");
					writer.write(String.valueOf(i));
					writer.write(":");
					writer.write(String.valueOf(values[i]));
				}
			}
			writer.write("\n");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	public void load(BufferedReader reader) {
		try {
			String line = reader.readLine().trim();
			String[] entries = line.split(" ");
			assert Integer.parseInt(entries[0]) == values.length; 
			Arrays.fill(values, 0.f);
			for(int i = 1;i < entries.length;i++){
				String[] weight = entries[i].split(":");
				assert Integer.parseInt(weight[0]) < values.length;
				values[Integer.parseInt(weight[0])] = Float.parseFloat(weight[1]);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void add(WeightVector dv) {
		add(1., dv);
	}
	
	public void add(double w, WeightVector weight) {
		assert values.length == weight.values.length;
		assert Math.abs(w) >= Double.MIN_VALUE;
		for(int i = 0;i < values.length;i++){
			values[i] += weight.values[i] * w;
		}
	}

	public void scale(double d) {
		for(int i = 0;i < values.length;i++){
			if(Math.abs(values[i]) >= Double.MIN_VALUE){
				values[i] *= d;
			}else{
				values[i] = 0.f;
			}
		}
	}

	public double[] get() {
		return values;
	}

	public void fill(double d) {
		Arrays.fill(values, 1.0);		
	}
}