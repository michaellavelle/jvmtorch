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

import org.jvmtorch.torch.Tensor;
import org.jvmtorch.torch.TensorOperations;
import org.jvmpy.python.Tuple;

public interface Functional<T extends TensorOperations<T>> {

	Tensor<T> relu(Tensor<T> input);
	
	Tensor<T> max_pool2d(Tensor<T> input, Tuple<Integer> tuple);
	
	Tensor<T> max_pool2d(Tensor<T> input, int i);

	Tensor<T> linear(Tensor<T> input, Parameter<T> weight, Parameter<T> bias);

}
