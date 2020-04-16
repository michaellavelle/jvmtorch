/*
 * Copyright 2020 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */
package org.jvmtorch.impl;

import java.util.List;

import org.jvmpy.python.Tuple;
import org.jvmtorch.torch.GradFunction;
import org.jvmtorch.torch.Size;
import org.jvmtorch.torch.Tensor;
import org.jvmtorch.torch.TensorData;
import org.jvmtorch.torch.TensorOperation;
import org.jvmtorch.torch.Torch;

public class TensorAdapter implements Tensor {

	protected Tensor adapted;
	
	public TensorAdapter(Tensor adapted) {
		this.adapted = adapted;
	}
	
	@Override
	public Tensor get() {
		return adapted.get();
	}

	public Tensor getTensor() {
		return adapted;
	}

	@Override
	public Tensor mul(float value) {
		return adapted.mul(value);
	}

	@Override
	public Tensor add(float value) {
		return adapted.add(value);
	}

	@Override
	public Tensor add(Tensor other) {
		return adapted.add(other);
	}

	@Override
	public Tensor mean() {
		return adapted.mean();
	}

	@Override
	public Tensor mul(Tensor other) {
		return adapted.mul(other);
	}

	@Override
	public int numel() {
		return adapted.numel();
	}

	@Override
	public Tensor mul_(Tensor other) {
		adapted.mul_(other);
		return this;
	}

	@Override
	public Tensor sub_(Tensor other) {
		adapted.sub_(other);
		return this;
	}

	@Override
	public Tensor add_(Tensor other) {
		adapted.add_(other);
		return this;
	}

	@Override
	public Tensor matmul(Tensor other) {
		return adapted.matmul(other);
	}

	@Override
	public Tensor t() {
		return adapted.t();
	}

	@Override
	public Size size() {
		return adapted.size();
	}

	@Override
	public void backward(Tensor gradient) {
		adapted.backward(gradient);
	}

	@Override
	public void backward() {
		adapted.backward();		
	}

	@Override
	public Tensor grad_fn_(GradFunction grad_fn) {
		adapted.grad_fn_(grad_fn);
		return this;
	}

	@Override
	public boolean requires_grad() {
		return adapted.requires_grad();
	}

	@Override
	public Tensor grad_(Tensor grad) {
		adapted.grad_(grad);
		return this;
	}

	@Override
	public float[] getDataAsFloatArray() {
		return adapted.getDataAsFloatArray();
	}

	@Override
	public Tensor grad() {
		return adapted.grad();
	}

	@Override
	public GradFunction grad_fn() {
		return adapted.grad_fn();
	}

	@Override
	public TensorData toTensorData() {
		return adapted.toTensorData();
	}

	@Override
	public Tensor requires_grad_(boolean requires_grad) {
		adapted.requires_grad_(requires_grad);
		return this;
	}

	@Override
	public Tensor withNextFunctions(String name, List<TensorOperation<Tensor>> operations,
			Tuple<Tuple<GradFunction>> nextFunctions) {
		adapted.withNextFunctions(name, operations, nextFunctions);
		return this;
	}

	@Override
	public Tensor performUnaryMappingOperation(String newTensorName,
			TensorOperation<TensorData> operation, TensorOperation<Tensor> backwardOp) {
		return adapted.performUnaryMappingOperation(newTensorName, operation, backwardOp);
	}

	@Override
	public Tensor view(int i, int num_flat_features) {
		return adapted.view(i, num_flat_features);
	}

	@Override
	public Tuple<String> names() {
		return adapted.names();
	}

	@Override
	public Tensor names_(Tuple<String> names) {
		adapted.names_(names);
		return this;
	}

	@Override
	public Tensor size_(Size size) {
		adapted.size_(size);
		return this;
	}

	@Override
	public Torch torch() {
		return adapted.torch();
	}

	@Override
	public Tensor view(Size size) {
		return adapted.view(size);
	}
}
