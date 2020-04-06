package org.jvmtorch.impl;

import org.jvmtorch.torch.*;
import org.jvmpy.python.Tuple;
import org.jvmtorch.torch.*;

import java.util.List;

public class TensorAdapter<T extends TensorOperations<T>> implements Tensor<T> {

	protected Tensor<T> adapted;
	
	public TensorAdapter(Tensor<T> adapted) {
		this.adapted = adapted;
	}
	
	@Override
	public Tensor<T> get() {
		return adapted.get();
	}

	public Tensor<T> getTensor() {
		return adapted;
	}

	@Override
	public Tensor<T> mul(float value) {
		return adapted.mul(value);
	}

	@Override
	public Tensor<T> add(float value) {
		return adapted.add(value);
	}

	@Override
	public Tensor<T> add(Tensor<T> other) {
		return adapted.add(other);
	}

	@Override
	public Tensor<T> mean() {
		return adapted.mean();
	}

	@Override
	public Tensor<T> mul(Tensor<T> other) {
		return adapted.mul(other);
	}

	@Override
	public int numel() {
		return adapted.numel();
	}

	@Override
	public Tensor<T> mul_(Tensor<T> other) {
		adapted.mul_(other);
		return this;
	}

	@Override
	public Tensor<T> sub_(Tensor<T> other) {
		adapted.sub_(other);
		return this;
	}

	@Override
	public Tensor<T> add_(Tensor<T> other) {
		adapted.add_(other);
		return this;
	}

	@Override
	public Tensor<T> matmul(Tensor<T> other) {
		return adapted.matmul(other);
	}

	@Override
	public Tensor<T> transpose() {
		return adapted.transpose();
	}

	@Override
	public Tensor<T> t() {
		return adapted.t();
	}

	@Override
	public Size size() {
		return adapted.size();
	}

	@Override
	public void backward(Tensor<T> gradient) {
		adapted.backward(gradient);
	}

	@Override
	public void backward() {
		adapted.backward();		
	}

	@Override
	public Tensor<T> grad_fn_(GradFunction<T> grad_fn) {
		adapted.grad_fn_(grad_fn);
		return this;
	}

	@Override
	public boolean requires_grad() {
		return adapted.requires_grad();
	}

	@Override
	public Tensor<T> grad_(Tensor<T> grad) {
		adapted.grad_(grad);
		return this;
	}

	@Override
	public float[] getDataAsFloatArray() {
		return adapted.getDataAsFloatArray();
	}

	@Override
	public Tensor<T> grad() {
		return adapted.grad();
	}

	@Override
	public GradFunction grad_fn() {
		return adapted.grad_fn();
	}

	@Override
	public T toTensorOperations() {
		return adapted.toTensorOperations();
	}

	@Override
	public Tensor<T> requires_grad_(boolean requires_grad) {
		adapted.requires_grad_(requires_grad);
		return this;
	}

	@Override
	public Tensor<T> withNextFunctions(String name, List<TensorOperation<Tensor<T>>> operations,
			Tuple<Tuple<GradFunction<T>>> nextFunctions) {
		adapted.withNextFunctions(name, operations, nextFunctions);
		return this;
	}

	@Override
	public Tensor<T> performUnaryMappingOperation(String newTensorName,
												  TensorOperation<T> operation, TensorOperation<Tensor<T>> backwardOp) {
		return adapted.performUnaryMappingOperation(newTensorName, operation, backwardOp);
	}

	@Override
	public Tensor<T> view(int i, int num_flat_features) {
		return adapted.view(i, num_flat_features);
	}

	@Override
	public float[] data() {
		return getDataAsFloatArray();
	}
}
