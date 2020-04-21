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

import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.jvmpy.python.OrderedDict;
import org.jvmpy.python.PythonClass;
import org.jvmtorch.nn.functional.Functional;
import org.jvmtorch.torch.Size;
import org.jvmtorch.torch.Tensor;
import org.jvmtorch.torch.Torch;


public abstract class Module<M> extends PythonClass<M> implements IModule {

	protected Functional F; 
	protected NN nn;
	protected OrderedDict<Parameter> moduleParameters;
	protected OrderedDict<IModule> subModules;
	protected String alias;

	protected Torch torch;

	public Module(NN nn) {
		this.F = nn.f();
		this.nn = nn;
		this.torch = nn.torch();
	}

	protected Parameter Parameter(Size size) {
		return nn.Parameter(size);
	}
	
	protected Parameter Parameter(Tensor tensor) {
		return nn.Parameter(tensor);
	}
	
	protected Size Size(int... sizes) {
		return torch.Size(sizes);
	}
	
	protected Size Size(Size... sizes) {
		return torch.Size(sizes);
	}
	
	public M alias_(String alias) {
		this.alias = alias;
		return self;
	}
	
	public String alias() {
		return alias;
	}
	
	protected OrderedDict<IModule> getSubModules() {
		if (subModules == null) {
			subModules = new OrderedDict<>(IModule.class);
			for (Entry<String, IModule> field : getFields(IModule.class)) {
				subModules.add(new ImmutablePair<>(field.getKey(), field.getValue()));
			}
		}
		return subModules;
	}
	
	protected OrderedDict<Parameter> getModuleParameters() {
		if (moduleParameters == null) {
			moduleParameters = new OrderedDict<>(Parameter.class);
			for (Map.Entry<String, Parameter> field : getFields(Parameter.class)) {
				moduleParameters.add(new ImmutablePair<>(field.getKey(), field.getValue()));
			}
		}
		return moduleParameters;
	}

	@Override
	public void zero_grad() {
		parameters().stream().map(p -> p.getRight()).forEach(p -> p.zero_grad());
	}

	@Override
	public Tensor apply(Tensor tensor) {
		return forward(tensor);
	}

	@Override
	public OrderedDict<Parameter> parameters() {
		OrderedDict<Parameter> parameters =new OrderedDict<>(Parameter.class);
		getModuleParameters().forEach(p -> parameters.add(p));
		getSubModules().forEach(m -> parameters.addAll(m.getRight().parameters()));
		return parameters;
	}

	@Override
	public String toString() {

		StringBuilder stringBuilder = new StringBuilder();
		stringBuilder.append(this.getClass().getSimpleName() + "(");
		List<Pair<String, IModule>> subModules = getSubModules();
		if (!subModules.isEmpty()) {
			stringBuilder.append("\n");
			for (Pair<String, ? extends IModule> layer : subModules) {
				stringBuilder.append("\t(");
				stringBuilder.append(layer.getKey());
				stringBuilder.append("): ");
				stringBuilder.append(layer.getValue().toString());
				stringBuilder.append("\n");
			}
		}
		//stringBuilder.append(attributes.toString());
		stringBuilder.append(")");

		return stringBuilder.toString();
	}

}
