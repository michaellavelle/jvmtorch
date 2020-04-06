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

import org.jvmpy.python.Tuple;

import java.util.List;

public interface Tensor<T extends TensorOperations<T>> extends TensorOperations<Tensor<T>> {

	void backward(Tensor<T> gradient);
	void backward();

	Tensor<T> grad_fn_(GradFunction<T> grad_fn);

	boolean requires_grad();

	Tensor<T> grad_(Tensor<T> grad);


	float[] getDataAsFloatArray();
	
	Tensor<T> grad();
	
	GradFunction<T> grad_fn();

	T toTensorOperations();

	//String name();

	Tensor<T> requires_grad_(boolean requires_grad);

	public Tensor<T> withNextFunctions(String name, List<TensorOperation<Tensor<T>>> operations, Tuple<Tuple<GradFunction<T>>> nextFunctions);

	//Tensor performUnaryMappingOperation(String newTensorName, String operationName, UnaryOperator<Tensor> operation, UnaryOperator<Tensor> backwardOp);

	Tensor<T> performUnaryMappingOperation(String newTensorName, TensorOperation<T> operation, TensorOperation<Tensor<T>> backwardOp);

	
	Tensor<T> view(int i, int num_flat_features);

}
