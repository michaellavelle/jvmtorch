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

import java.util.function.BiFunction;

public class TensorDataConverter<A extends TensorOperations<A>> {

	
	protected BiFunction<float[], Size, A> adaptedFactory;
	protected Class<A> adaptedClass;
	
	public TensorDataConverter(Class<A> adaptedClass, BiFunction<float[], Size, A> adaptedFactory) {
		this.adaptedClass = adaptedClass;
		this.adaptedFactory = adaptedFactory;
	}
	
	public TensorData createTensorDataFromTensorOperations(A adapted) {
		return new TensorDataAdapter<>(adapted, this);
	}
	
	public A createTensorOperationsFromTensorData(TensorData tensorData) {
		if (tensorData instanceof TensorDataAdapter) {
			TensorDataAdapter<?> adapter = (TensorDataAdapter<?>)tensorData;
			if (adaptedClass.isAssignableFrom(adapter.adapted.getClass())) {
				return adaptedClass.cast(adapter.adapted);
			}
		}
		return adaptedFactory.apply(tensorData.getDataAsFloatArray(), tensorData.size());
	}
}
