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

public abstract class Conv2d<M extends Conv2d<M, T>, T extends TensorOperations<T>> extends Module<M, T> implements IModule<M, T> {

	protected int in_channels;
	protected int out_channels;
	protected int kernel_size;
	protected Parameter<T> weight;
	protected Parameter<T> bias;

	public Conv2d(NN<T> nn, int in_channels,
				  int out_channels,
				  int kernel_size) {
		super(nn);
		self.in_channels = in_channels;
		self.out_channels = out_channels;
		self.kernel_size = kernel_size;
		self.weight = Parameter(self().out_channels, self().in_channels * self().kernel_size * self().kernel_size);
		self.bias = Parameter(self().out_channels, 1);
	}

	@Override
	public void zero_grad() {
		self().bias.grad_(null);
		self().weight.grad_(null);
	}

	public Parameter<T> bias() {
		return bias;
	}

	public Parameter<T> weight() {
		return weight;
	}
}
