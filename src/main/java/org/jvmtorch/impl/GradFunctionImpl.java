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

import org.jvmtorch.torch.GradFunction;
import org.jvmtorch.torch.Tensor;
import org.jvmtorch.torch.TensorOperation;
import org.jvmtorch.torch.TensorOperations;
import org.jvmpy.python.Tuple;

import java.util.List;

public class GradFunctionImpl<T extends TensorOperations<T>> extends GradFunction<T> {

	private Tensor<T> variable;
	
	public GradFunctionImpl(String name, List<TensorOperation<Tensor<T>>> operations, Tensor<T> variable, Tuple<Tuple<GradFunction<T>>> next_functions) {
		super(name, operations, next_functions);
		this.variable = variable;
	}

	@Override
	public String toString() {
		return "<" + name() + " object>";
	}

	public Tensor<T> variable() {
		return variable;
	}

}
