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

import org.jvmtorch.nn.IModule;
import org.jvmtorch.nn.NN;
import org.jvmtorch.nn.Parameter;
import org.jvmtorch.nn.functional.Functional;
import org.jvmtorch.nn.modules.container.Sequential;
import org.jvmtorch.torch.Size;
import org.jvmtorch.torch.Tensor;
import org.jvmtorch.torch.Torch;


public abstract class NNImpl implements NN {

	protected Torch torch;
	protected Functional f;

	public NNImpl(Torch torch, Functional f) {
		this.torch = torch;
		this.f = f;
	}

	public Torch torch() {
		return torch;
	}
	
	public Functional f() {
		return f;
	}

	public Parameter Parameter(Tensor tensor) {
		return new ParameterImpl(tensor);
	}

	@Override
	public Parameter Parameter(Size size) {
		return new ParameterImpl(torch.randn(size).mul(0.01f));
	}

	@Override
	public Sequential Sequential(IModule... modules) {
		return new Sequential(this, modules);
	}
	
	
}
