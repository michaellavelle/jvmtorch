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

import org.jvmtorch.torch.Tensor;


public abstract class Linear<M extends Linear<M>> extends Module<M> implements IModule {

	protected int in_features;
	protected int out_features;
	protected Parameter weight;
	protected Parameter bias;

	public Linear(NN nn, int in_features, int out_features) {
		super(nn);
		self.in_features = in_features;
		self.out_features = out_features;
		self.weight = Parameter(initialWeights()
				.names_(tuple("output_feature", "input_feature")));
		self.bias = Parameter(initialBias().names_(tuple("example", "output_feature")));		
	}
	
	private Tensor initialBias() {
		return torch.randn(torch.Size(1, self.out_features)).mul((float) Math.sqrt(1f / in_features));
	}
	
	private Tensor initialWeights() {
		return torch.randn(torch.Size(self.out_features, self.in_features)).mul((float) Math.sqrt(1f / in_features));
	}

	@Override
	public void zero_grad() {
		self.bias.grad_(null);
		self.weight.grad_(null);
	}
	
	public Parameter bias() {
		return bias;
	}

	public Parameter weight() {
		return weight;
	}
}
