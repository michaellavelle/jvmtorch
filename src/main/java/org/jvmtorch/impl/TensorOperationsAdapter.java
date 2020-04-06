package org.jvmtorch.impl;

import org.jvmtorch.torch.Size;
import org.jvmtorch.torch.TensorOperations;

public abstract class TensorOperationsAdapter<T, A extends TensorOperations<T>> implements TensorOperations<T> {

	protected A adapted; 
	
	public TensorOperationsAdapter(A adapted) {
		this.adapted = adapted;
	}
	
	@Override
	public T get() {
		return adapted.get();
	}

	@Override
	public T mul(float value) {
		return adapted.mul(value);
	}

	@Override
	public T add(float value) {
		return adapted.add(value);
	}

	@Override
	public T mul(T other) {
		return adapted.mul(other);
	}

	@Override
	public int numel() {
		return adapted.numel();
	}

	@Override
	public T add(T other) {
		return adapted.add(other);
	}

	@Override
	public T mean() {
		return adapted.mean();
	}

	@Override
	public T mul_(T other) {
		adapted.mul_(other);
		return self();
	}

	protected abstract T self();

	@Override
	public T sub_(T other) {
		adapted.sub_(other);
		return self();
	}

	@Override
	public T add_(T other) {
		adapted.add_(other);
		return self();
	}

	@Override
	public T matmul(T other) {
		return adapted.matmul(other);
	}

	@Override
	public T transpose() {
		return adapted.transpose();
	}

	@Override
	public T t() {
		return adapted.t();
	}

	@Override
	public Size size() {
		return adapted.size();
	}

	@Override
	public float[] data() {
		return adapted.data();
	}
}
