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
package org.jvmtorch.nn.modules.container;

import java.util.Arrays;
import java.util.stream.Collectors;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.jvmpy.python.OrderedDict;
import org.jvmtorch.nn.IModule;
import org.jvmtorch.nn.Module;
import org.jvmtorch.nn.NN;
import org.jvmtorch.torch.Tensor;


public class Sequential extends Module<Sequential> implements IModule {

	protected IModule[] initialModules;

	public Sequential(NN nn, IModule... modules) {
		super(nn);
		this.initialModules = modules;
	}

	@Override
	public Tensor forward(Tensor input) {
		var in_progress = input;
		for (Pair<String, IModule> module : getSubModules()) {
			in_progress = module.getRight().forward(in_progress);
		}
		var output = in_progress; 
		return output;
	}
	
	@Override
	protected OrderedDict<IModule> getSubModules() {
		OrderedDict<IModule> modules = new OrderedDict<IModule>(IModule.class);
		modules.addAll(Arrays.stream(this.initialModules).map(m -> new ImmutablePair<>("somename", m)).collect(Collectors.toList()));
		return modules;
	}

	@Override
	protected Sequential self() {
		return this;
	}

	@Override
	public void zero_grad() {
		for (Pair<String, IModule> module : getSubModules()) {
			module.getRight().zero_grad();
		}
	}
	
}
