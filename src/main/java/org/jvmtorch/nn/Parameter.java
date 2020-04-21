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

import java.util.List;

import org.jvmpy.python.Tuple;
import org.jvmtorch.torch.GradFunction;
import org.jvmtorch.torch.Size;
import org.jvmtorch.torch.Tensor;
import org.jvmtorch.torch.TensorData;
import org.jvmtorch.torch.TensorOperation;

public interface Parameter extends Tensor {

	void zero_grad();

	Tensor data();
	
	Tensor data_(Tensor data);

	@Override
	Parameter grad_(Tensor grad);

	@Override
	Parameter requires_grad_(boolean requires_grad);
	
	@Override
	Parameter withNextFunctions(String name, List<TensorOperation<Tensor, Size>> operations,
			Tuple<Tuple<GradFunction>> nextFunctions);
	
	@Override
	Tensor performUnaryMappingOperation(TensorOperation<TensorData, Size> operation, TensorOperation<Tensor, Size> backwardOp);

	@Override
	Parameter mul_(Tensor other);

	@Override
	Parameter sub_(Tensor other);

	@Override
	Parameter add_(Tensor other);

	@Override
	Tensor t();

	
}
