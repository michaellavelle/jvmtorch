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
	public T t() {
		return adapted.t();
	}

	@Override
	public Size size() {
		return adapted.size();
	}

	@Override
	public float[] getDataAsFloatArray() {
		return adapted.getDataAsFloatArray();
	}
}
