package model;

import inference.State;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.List;

import com.google.common.io.Files;
import config.Parameters;

import data.Instance;
import data.Label;

// This is an implementaion of SCW.
// Jialei Wang, Peillin Zhao, Steven C.H. Hoi
// Exact Soft Confidence-Weighted Learning
// ICML2012

public final class SCWModel extends Model {
	private WeightVector covariance;
	private final double eta = 0.75;
	private final double phi;
	private final double phiSquared;
	private final double psi;
	private final double gsi;
	private final double C;

	private static final double normsinv(double p){
		assert 0.5 < p && p < 1; 
		double a1 = -39.6968302866538, a2 = 220.946098424521, a3 = -275.928510446969;
		double a4 = 138.357751867269, a5 = -30.6647980661472, a6 = 2.50662827745924;
		double b1 = -54.4760987982241, b2 = 161.585836858041, b3 = -155.698979859887;
		double b4 = 66.8013118877197, b5 = -13.2806815528857, c1 = -7.78489400243029E-03;
		double c2 = -0.322396458041136, c3 = -2.40075827716184, c4 = -2.54973253934373;
		double c5 = 4.37466414146497, c6 = 2.93816398269878, d1 = 7.78469570904146E-03;
		double d2 = 0.32246712907004, d3 = 2.445134137143, d4 = 3.75440866190742;
		double p_low = 0.02425, p_high = 1 - p_low;
		double q, r;
		double retVal;
		if (p < p_low){
			q = Math.sqrt(-2 * Math.log(p));
			retVal = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
		}else if (p <= p_high){
			q = p - 0.5;
			r = q * q;
			retVal = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1);
		}else{
			q = Math.sqrt(-2 * Math.log(1 - p));
			retVal = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
		}
		return retVal;
	}

	public SCWModel(Parameters params, FeatureGenerator fg) {		
		super(params, fg);
		covariance = new WeightVector(params);
		covariance.fill(1.);
		phi = normsinv(eta);
		assert phi > 0.: phi;
		phiSquared = phi * phi;
		psi = 1. + phiSquared / 2.;
		assert psi > 0.: psi;
		gsi = 1. + phiSquared;
		assert gsi > 0.: gsi;
		C = params.getLambda();
	}
	

	public double getConfidence(SparseFeatureVector fv){
		double c = 0.0;
		for(VectorInfo vector:fv.getFeatureVectors()){
			int[] index = vector.getSv().getIndex();
			double[] grad = vector.getSv().getData();
			double[] cov = covariance.get();
			int size = vector.getSv().getUsed();
			assert size > 0.;
			double localC = 0.;
			for(int i = 0;i < size;i++){
				localC += cov[index[i]] * grad[i] * grad[i];
			}
			double scale = vector.getWeight();
			c += localC * scale * scale;
		}
		return c;
	}

	private void update(SparseFeatureVector fv, double m_t, double phi) {
		double v_t = getConfidence(fv);
		double alpha_tmp = -m_t*psi;
		alpha_tmp += Math.sqrt(Math.pow(m_t * phiSquared / 2, 2) + v_t * phiSquared * gsi);
		alpha_tmp /= v_t * gsi;
		double alpha_t = Math.min(C, Math.max(0., alpha_tmp));
		if(alpha_t == 0.){
			return;
		}
		assert alpha_t > 0. && !Double.isNaN(alpha_t) && !Double.isInfinite(alpha_t) : alpha_t;
		double avp = alpha_t * v_t * phi;
		double sqrt_u_t = -avp + Math.sqrt(avp*avp+4*v_t);
		sqrt_u_t = Math.abs(sqrt_u_t / 2.);
		double beta_t = alpha_t * phi / (sqrt_u_t+avp);
		assert beta_t > 0. && !Double.isNaN(beta_t) && !Double.isInfinite(beta_t) : beta_t;
		double[] w = weight.get();
		double[] wdiff = null;
		if(params.getUseAveraging()){
			wdiff = weightDiff.get();
		}
		for(VectorInfo vector:fv.getFeatureVectors()){
			int[] index = vector.getSv().getIndex();
			double[] grad = vector.getSv().getData();
			double[] cov = covariance.get();
			double scale = vector.getWeight();
			int size = vector.getSv().getUsed();
			for(int i = 0;i < size;i++){
				int index_i = index[i];
				double cov_i_grad_i = cov[index_i] * grad[i] * scale;
				double diff = alpha_t * cov_i_grad_i;
				w[index_i] += diff;
				if(params.getUseAveraging()){
					wdiff[index_i] += trainStep * diff;
				}
				cov[index_i] -= beta_t * cov_i_grad_i * cov_i_grad_i;
			}
		}
		trainStep++;
	}
		
	@Override
	public void update(List<State> updates) {
		SparseFeatureVector fv = new SparseFeatureVector(params);
		double gradSum = 0.;
		for(State yStarState:updates){
			Instance instance = yStarState.getInstance();	
			Label yGold = instance.getGoldLabel();
			if(yStarState.isCorrect()){
				continue;
			}
			assert !yGold.equals(yStarState.getLabel()) : yGold.size();
			SparseFeatureVector diffFv = yStarState.getDiffFv(); 
			double diffScore = evaluate(diffFv, false) - yStarState.getMargin();
			if(Math.max(0, -diffScore) > 0.){
				fv.add(diffFv);
				gradSum += diffScore;
			}
		}
		fv.compact();
		if(fv.size() == 0){
			return;
		}
		if(params.getMiniBatch() != 1){
			fv.scale(1./params.getMiniBatch());
			gradSum /= params.getMiniBatch();
		}
		update(fv, gradSum, phi);
	}

	@Override
	public void save(String filename){
		try {
			BufferedWriter writer = new BufferedWriter(Files.newWriterSupplier(new File(filename), Charset.forName("UTF-8")).getOutput());
			writer.write(String.valueOf(trainStep));
			writer.write("\n");
			weight.save(writer);
			if(params.getUseAveraging()){
				weightDiff.save(writer);
				aveWeight.save(writer);
			}
			covariance.save(writer);
			params.saveModelParameters(writer);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	public void load(String filename){
		try {
			BufferedReader reader = new BufferedReader(Files.newReaderSupplier(new File(filename), Charset.forName("UTF-8")).getInput());
			trainStep = Integer.parseInt(reader.readLine().trim());
			weight.load(reader);
			if(params.getUseAveraging()){
				weightDiff.load(reader);
				aveWeight.load(reader);
			}
			covariance.load(reader);
			params.loadModelParameters(reader);
			reader.close();
			fg.init();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public double getPhi() {
		return phi;
	}

}
