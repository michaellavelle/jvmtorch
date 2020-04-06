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
package org.jvmtorch.nn;

import org.jvmtorch.torch.GradFunction;
import org.jvmtorch.torch.Tensor;
import org.jvmtorch.torch.TensorOperation;
import org.jvmtorch.torch.TensorOperations;
import org.jvmpy.python.Tuple;

import java.util.List;

public interface Parameter<T extends TensorOperations<T>> extends Tensor<T> {

	void zero_grad();

	Tensor<T> getTensor();

	@Override
	Parameter<T> grad_(Tensor<T> grad);

	@Override
	Parameter<T> requires_grad_(boolean requires_grad);

	@Override
	Parameter<T> withNextFunctions(String name, List<TensorOperation<Tensor<T>>> tensorOperations, Tuple<Tuple<GradFunction<T>>> nextFunctions);

	@Override
	Tensor<T> performUnaryMappingOperation(String newTensorName, TensorOperation<T> operation, TensorOperation<Tensor<T>> backwardOp);

	@Override
	Parameter<T> mul_(Tensor<T> other);

	@Override
	Parameter<T> sub_(Tensor<T> other);

	@Override
	Parameter<T> add_(Tensor<T> other);

	@Override
	Tensor<T> transpose();
	
}
