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
package org.jvmtorch.nn.modules;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.jvmpy.python.OrderedDict;
import org.jvmpy.python.PythonClass;
import org.jvmtorch.nn.Functional;
import org.jvmtorch.nn.IModule;
import org.jvmtorch.nn.NN;
import org.jvmtorch.nn.Parameter;
import org.jvmtorch.torch.Tensor;
import org.jvmtorch.torch.TensorOperations;
import org.jvmtorch.torch.Torch;

import java.util.List;
import java.util.Map;


public abstract class  Module<M, I extends TensorOperations<I>> extends PythonClass<M> implements IModule<M, I> {

	protected Functional<I> F;
	protected NN<I> nn;
	protected OrderedDict<Parameter<I>> moduleParameters;
	protected OrderedDict<IModule<?, I>> subModules;

	protected Torch<I> torch;

	public Module(NN<I> nn) {
		this.F = nn.f();
		this.nn = nn;
		this.torch = nn.torch();
	}

	protected Parameter <I> Parameter(int...dims) {
		return nn.Parameter(dims);
	}
	
	protected OrderedDict<IModule<?, I>> getSubModules() {
		if (subModules == null) {
			subModules = new OrderedDict<>(IModule.class);
			for (Map.Entry<String, IModule> field : getFields(IModule.class).entrySet()) {
				subModules.add(new ImmutablePair<>(field.getKey(), field.getValue()));
			}
		}
		return subModules;
	}
	
	@SuppressWarnings("unchecked")
	protected OrderedDict<Parameter<I>> getModuleParameters() {
		if (moduleParameters == null) {
			moduleParameters = new OrderedDict<>(Parameter.class);
			for (Map.Entry<String, Parameter> field : getFields(Parameter.class).entrySet()) {
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
	public Tensor<I> apply(Tensor<I> tensor) {
		return forward(tensor);
	}

	@Override
	public OrderedDict<Parameter<I>> parameters() {
		OrderedDict<Parameter<I>> parameters =new OrderedDict<>(Parameter.class);
		getModuleParameters().forEach(p -> parameters.add(p));
		getSubModules().forEach(m -> parameters.addAll(m.getRight().parameters()));
		return parameters;
	}

	@Override
	public String toString() {

		StringBuilder stringBuilder = new StringBuilder();
		stringBuilder.append(this.getClass().getSimpleName() + "(");
		List<Pair<String, IModule<?, I>>> subModules = getSubModules();
		if (!subModules.isEmpty()) {
			stringBuilder.append("\n");
			for (Pair<String, ? extends IModule<?, ?>> layer : subModules) {
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
