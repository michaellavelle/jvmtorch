package org.jvmtorch.torch;

public interface FullTensorOperations<T> extends TensorOperations<T> {

	T mul(float value);

	T add(float value);
	
	T sub(float value);


	T mul(T other);
	
	T div(T other);
	
	T sub(T other);

	int numel();
	
	T sum();

	T add(T other);

	T mean();
	
	T norm();

	T mul_(T other);
	
	T columnSums();
	
	T rowSums();

	T cloneTensor();


	T sub_(T other);

	T add_(T other);

	T matmul(T other);

	T t();

	Size size();
	
	T size_(Size size);
		
	T view(Size size);
	
	void close();

}
