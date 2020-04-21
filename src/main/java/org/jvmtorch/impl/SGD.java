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

import org.jvmpy.python.OrderedDict;
import org.jvmtorch.nn.Parameter;
import org.jvmtorch.torch.Torch;
import org.jvmtorch.torch.optim.Optimiser;

public class SGD extends OptimiserImpl<SGD> implements Optimiser {

	protected Float learning_rate;
	protected OrderedDict<Parameter> parameters;
	protected Torch torch;

	public SGD(Torch torch, OrderedDict<Parameter> parameters, Number learning_rate) {
		this.learning_rate = learning_rate == null ? null : learning_rate.floatValue();
		this.parameters = parameters;
		this.torch = torch;
	}

	@Override
	public SGD self() {
		return this;
	}

	@Override
	public void step() {
		for (int i = 0; i < parameters.size(); i++) {
			Parameter param = parameters.get(i).getRight();
			param.requires_grad_(false);
			if (param.grad() != null) {
				var delta = param.grad().mul(learning_rate.floatValue() / (1000f));
				param.sub_(delta);
			}
		}
	}

	@Override
	public void zero_grad() {
		for (int i = 0; i < parameters.size(); i++) {
			Parameter param = parameters.get(i).getRight();
			if (param.grad() != null) {
				param.data().grad_(null);
			}
			param.requires_grad_(true);
		}

	}

	@Override
	public String toString() {
		return "SGD [lr=" + learning_rate + "]";
	}
}
