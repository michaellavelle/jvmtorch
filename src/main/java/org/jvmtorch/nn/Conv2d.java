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


import static org.jvmpy.python.Python.tuple;

public abstract class Conv2d<M extends Conv2d<M>> extends Module<M> implements IModule {

	protected int in_channels;
	protected int out_channels;
	protected int kernel_size;
	protected Parameter weight;
	protected Parameter bias;

	public Conv2d(NN nn, int in_channels,
				  int out_channels,
				  int kernel_size) {
		super(nn);
		self.in_channels = in_channels;
		self.out_channels = out_channels;
		self.kernel_size = kernel_size;
		
		self.weight = Parameter(torch.Size(torch.Size(self().out_channels), 
				torch.Size(self().in_channels, self().kernel_size, self().kernel_size)
				).names_(tuple("output_depth", "input_depth", "filter_height", "filter_width")));
	
		self.bias = Parameter(torch.Size(self().out_channels, 1).names_(tuple("output_depth", "None")));
	}

	@Override
	public void zero_grad() {
		self().bias.grad_(null);
		self().weight.grad_(null);
	}

	public Parameter bias() {
		return bias;
	}

	public Parameter weight() {
		return weight;
	}
}
