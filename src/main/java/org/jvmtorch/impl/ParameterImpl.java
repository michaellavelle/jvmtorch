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

import static org.jvmpy.python.Python.True;

import java.util.List;

import org.jvmpy.python.Tuple;
import org.jvmtorch.nn.Parameter;
import org.jvmtorch.torch.GradFunction;
import org.jvmtorch.torch.Size;
import org.jvmtorch.torch.Tensor;
import org.jvmtorch.torch.TensorOperation;

public class ParameterImpl extends TensorAdapter implements Parameter {
	
	public ParameterImpl(Tensor adapted) {
		super(adapted.requires_grad_(True));
	}

	@Override
	public void zero_grad() {
		grad_(null);
	}

	@Override
	public String toString() {
		return "Parameter containing:\n" + this.adapted.toString();
		//return "Parameter";
	}
	
	@Override
	public Parameter sub_(Tensor other) {
		super.sub_(other);
		return this;
	}

	@Override
	public Parameter mul_(Tensor other) {
		super.mul_(other);
		return this;
	}

	@Override
	public Parameter withNextFunctions(String name, List<TensorOperation<Tensor, Size>> tensorOperations, Tuple<Tuple<GradFunction>> nextFunctions) {
		super.withNextFunctions(name, tensorOperations, nextFunctions);
		return this;
	}

	@Override
	public Parameter requires_grad_(boolean requires_grad) {
		super.requires_grad_(requires_grad);
		return this;
	}

	@Override
	public Parameter grad_(Tensor grad) {
		super.grad_(grad);
		return this;
	}

	@Override
	public Parameter add_(Tensor other) {
		super.add_(other);
		return this;
	}

	@Override
	public Tensor data() {
		return adapted;
	}

	@Override
	public Tensor data_(Tensor data) {
		this.adapted = data;
		return this;
	}
}
