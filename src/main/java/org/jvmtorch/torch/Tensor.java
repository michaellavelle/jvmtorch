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

import java.util.List;

import org.jvmpy.python.Tuple;

public interface Tensor extends TensorOperations<Tensor> {

	void backward(Tensor gradient);
	
	void backward(Tensor gradient, boolean create_graph);
	
	void backward();

	Tensor grad_fn_(GradFunction grad_fn);

	boolean requires_grad();
	
	boolean create_graph();
	
	Tensor create_graph_(boolean create_graph);
	
	Tuple<String> names();
	
	Tensor names_(Tuple<String> names);

	Tensor grad_(Tensor grad);

	float[] getDataAsFloatArray();
	
	Tensor grad();
	
	float item();
	
	GradFunction grad_fn();

	TensorData toTensorData();
	
	Torch torch();

	Tensor requires_grad_(boolean requires_grad);

	public Tensor withNextFunctions(String name, List<TensorOperation<Tensor, Size>> operations, Tuple<Tuple<GradFunction>> nextFunctions);

	Tensor performUnaryMappingOperation(TensorOperation<TensorData, Size> operation, TensorOperation<Tensor, Size> backwardOp);

	Tensor view(int i, int num_flat_features);
	
	Tensor view(Size size);

}
