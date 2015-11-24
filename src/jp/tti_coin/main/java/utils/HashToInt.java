package utils;

import java.nio.charset.Charset;

import com.google.common.hash.HashFunction;
import com.google.common.hash.Hashing;

import config.Parameters;

public class HashToInt {
	private final HashFunction murmur3;
	private static final Charset UTF8 = Charset.forName("UTF-8");
	private final int filter;
	
	public HashToInt(Parameters params){
		this.murmur3 = Hashing.murmur3_32();
		this.filter = params.fvSize() - 1;
	}
	
	public int mapToInt(String key){
		return murmur3.hashString(key, UTF8).asInt() & filter;
	}
	

}
