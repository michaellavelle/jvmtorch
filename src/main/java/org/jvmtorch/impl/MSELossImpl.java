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
import org.jvmtorch.nn.modules.MSELoss;
import org.jvmtorch.torch.Tensor;


public abstract class MSELossImpl implements MSELoss {

	protected OrderedDict<Parameter> parameters;

	@Override
	public OrderedDict<Parameter> parameters() {
		return parameters;
	}

	@Override
	public void zero_grad() {
		if (parameters != null) {
			parameters.forEach(p -> p.getRight().zero_grad());
		}
	}

	@Override
	public Tensor apply(Tensor t, Tensor u) {
		return forward(this, t, u);
	}
}
