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
import org.jvmtorch.torch.TensorOperation;

public class GradFunctionImpl implements GradFunction {

	private String name;
	private List<TensorOperation<Tensor, Size>> operations;
	private Tuple<Tuple<GradFunction>> next_functions;
	private Tensor variable;
	
	public GradFunctionImpl(String name, List<TensorOperation<Tensor, Size>> operations, Tensor variable, Tuple<Tuple<GradFunction>> next_functions) {
		this.name = name;
		this.operations = operations;
		this.next_functions = next_functions;
		this.variable = variable;
	}

	@Override
	public String toString() {
		return "<" + name() + " object>";
	}

	public Tensor variable() {
		return variable;
	}

	@Override
	public List<TensorOperation<Tensor, Size>> operations() {
		return operations;
	}

	@Override
	public String name() {
		return name;
	}

	@Override
	public Tuple<Tuple<GradFunction>> next_functions() {
		return next_functions;
	}

}
