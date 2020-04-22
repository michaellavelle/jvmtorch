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
package org.jvmtorch.torch;

/**
 * Adapts a provider-specific TensorOperations instance of type A to a TensorData
 * instance for use by Tensor/SymbolicTensor.
 * 
 * @author Michael Lavelle
 *
 * @param <A> The type of the provider-specific TensorOperations instance
 */
public class TensorDataAdapter<A extends TensorOperations<A>> implements TensorData {

	/**
	 * The adapted provider-specifc TensorOperations instance
	 */
	protected A adapted; 
	
	/**
	 * The converter used to convert between TensorData and the provider-specific TensorOperations
	 */
	protected TensorDataConverter<A> tensorDataConverter;
	
	public TensorDataAdapter(A adapted,  TensorDataConverter<A> tensorDataConverter) {
		this.adapted = adapted;
		this.tensorDataConverter = tensorDataConverter;
	}
	
	/**
	 * @return the adapted provider-specifc TensorOperations instance.
	 */
	public A getAdapted() {
		return adapted;
	}
	
	private TensorData createTensorDataFromTensorOperations(A adapted) {
		return tensorDataConverter.createTensorDataFromTensorOperations(adapted);
	}
	
	private A createTensorOperationsFromTensorData(TensorData tensorData) {
		return tensorDataConverter.createTensorOperationsFromTensorData(tensorData);
	}
	
	@Override
	public TensorData get() {
		return createTensorDataFromTensorOperations(adapted.get());
	}

	@Override
	public TensorData mul(float value) {
		return createTensorDataFromTensorOperations(adapted.mul(value));
	}

	@Override
	public TensorData add(float value) {
		return createTensorDataFromTensorOperations(adapted.add(value));
	}
	
	@Override
	public TensorData sub(float value) {
		return createTensorDataFromTensorOperations(adapted.sub(value));
	}

	@Override
	public TensorData mul(TensorData other) {
		return createTensorDataFromTensorOperations(adapted.mul(createTensorOperationsFromTensorData(other)));
	}
	
	@Override
	public TensorData div(TensorData other) {
		return createTensorDataFromTensorOperations(adapted.div(createTensorOperationsFromTensorData(other)));
	}

	@Override
	public int numel() {
		return adapted.numel();
	}

	@Override
	public TensorData add(TensorData other) {
		return createTensorDataFromTensorOperations(adapted.add(createTensorOperationsFromTensorData(other)));
	}
	
	@Override
	public TensorData sub(TensorData other) {
		return createTensorDataFromTensorOperations(adapted.sub(createTensorOperationsFromTensorData(other)));
	}

	@Override
	public TensorData mean() {
		return createTensorDataFromTensorOperations(adapted.mean());
	}
	
	@Override
	public TensorData sum() {
		return createTensorDataFromTensorOperations(adapted.sum());
	}

	@Override
	public TensorData mul_(TensorData other) {
		adapted.mul_(createTensorOperationsFromTensorData(other));
		return self();
	}

	protected TensorData self() {
		return this;
	}

	@Override
	public TensorData sub_(TensorData other) {
		adapted.sub_(createTensorOperationsFromTensorData(other));
		return self();
	}

	@Override
	public TensorData add_(TensorData other) {
		adapted.add_(createTensorOperationsFromTensorData(other));
		return self();
	}

	@Override
	public TensorData matmul(TensorData other) {
		return createTensorDataFromTensorOperations(adapted.matmul(createTensorOperationsFromTensorData(other)));
	}

	@Override
	public TensorData t() {
		return createTensorDataFromTensorOperations(adapted.t());
	}

	@Override
	public Size size() {
		return adapted.size();
	}

	@Override
	public float[] getDataAsFloatArray() {
		return adapted.getDataAsFloatArray();
	}

	@Override
	public TensorData size_(Size size) {
		adapted.size_(size);
		return self();
	}

	@Override
	public TensorData view(int i, int j) {
		return createTensorDataFromTensorOperations(adapted.view(i, j));
	}

	@Override
	public TensorData view(Size size) {
		return createTensorDataFromTensorOperations((adapted.view(size)));
	}

	@Override
	public TensorData columnSums() {
		return createTensorDataFromTensorOperations((adapted.columnSums()));
	}
	
	@Override
	public TensorData rowSums() {
		return createTensorDataFromTensorOperations((adapted.rowSums()));
	}

	@Override
	public void close() {
		this.adapted = null;
		this.tensorDataConverter = null;
	}

	@Override
	public TensorData norm() {
		return createTensorDataFromTensorOperations((adapted.norm()));
	}

	@Override
	public TensorData cloneTensor() {
		return createTensorDataFromTensorOperations((adapted.cloneTensor()));
	}
}
