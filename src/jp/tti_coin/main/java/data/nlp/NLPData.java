package data.nlp;

import java.util.List;

import config.Parameters;
import data.Data;

public abstract class NLPData extends Data {
	protected List<Document> documents;

	public NLPData(Parameters params, String fileBase, boolean isTrain) {
		super(params, fileBase, isTrain);
		assert documents != null;
	}

	public List<Document> getDocuments() {
		return documents;
	}

	@Override
	protected abstract void load(String dirBase);

}
