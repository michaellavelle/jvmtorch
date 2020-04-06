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

import org.jvmtorch.nn.modules.Module;
import org.jvmtorch.torch.TensorOperations;

public abstract class Linear<M extends Linear<M, T>, T extends TensorOperations<T>> extends Module<M, T> implements IModule<M, T> {

	protected int in_features;
	protected int out_features;
	protected Parameter<T> weight;
	protected Parameter<T> bias;

	public Linear(NN<T> nn, int in_features, int out_features) {
		super(nn);
		self.in_features = in_features;
		self.out_features = out_features;
		self.weight = Parameter(self.out_features, self.in_features);
		self.bias = Parameter(1, self.out_features);
	}

	@Override
	public void zero_grad() {
		self.bias.grad_(null);
		self.weight.grad_(null);
	}
}
