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
package org.jvmpy.symbolictensors;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;

public class MappedSymbolicTensorImpl<S extends TensorDataContainer, T extends TensorDataContainer> implements MappedSymbolicTensor<S, T> {

	private String name;
	protected SymbolicTensor<S> init;
	private Function<S, T> mapper;
	private List<Operation<T>> operations;
	private Class<S> sClass;
	private Class<T> tClass;
	private int[] dimensions;
	private List<String> dimensionNames;

	protected MappedSymbolicTensorImpl(String name, Class<S> sClass, Class<T> tClass, int[] dims, List<String> dimensionNames) {
		this.name = name;
		this.operations = new ArrayList<>();
		this.sClass = sClass;
		this.tClass = tClass;
		this.dimensions = dims;
		this.dimensionNames = dimensionNames;
		if (dimensionNames != null && dimensionNames.size() != dimensions.length) {
			throw new IllegalArgumentException();
		}
	}
	
	public List<Operation<T>> getOperations() {
		return operations;
	}

	public MappedSymbolicTensorImpl(String name, SymbolicTensor<S> init, Function<S, T> mapper, Class<S> sClass, Class<T> tClass) {
		this(name, sClass, tClass, init.dimensions(), init.dimensionNames());
		init(init, mapper);
	}

	@Override
	public List<String> getAllOperationNames() {
		List<String> operationNames = new ArrayList<>();
		operationNames.addAll(init.getAllOperationNames());
		operationNames.addAll(getOperationNames());
		return operationNames;
	}

	@Override
	public void init(SymbolicTensor<S> init, Function<S, T> mapper) {
		this.init = init;
		this.mapper = mapper;
		this.dimensions = init.dimensions();
	}
	
	@Override
	public T evaluate() {
		T evaluation = getInput().get();
		for (Operation<T> operation : operations) {
			evaluation = operation.apply(evaluation);
		}
		return evaluation;
	}

	@Override
	public void performInlineOperation(Operation<T> operation) {
		this.operations.add(operation);
		TensorDimensionsContainer mappedDimensions = operation.dimensionsMapping().apply(this);
		this.dimensions = mappedDimensions.dimensions();
		this.dimensionNames = mappedDimensions.dimensionNames();
		if (dimensionNames != null && dimensionNames.size() != dimensions.length) {
				throw new IllegalArgumentException();
		}
	}

	@Override
	public SymbolicTensor<T> performUnaryMappingOperation(String newTensorName, Operation<T> operation) {
		SymbolicTensor<T> mappedTensor = new UnaryMappedSymbolicTensorImpl<T>(newTensorName, this);
		mappedTensor.performInlineOperation(operation);
		return mappedTensor;
	}

	@Override
	public String getName() {
		return name;
	}

	@Override
	public String getInputName() {
		if (init == null) {
			throw new IllegalStateException("Tensor '" + getName() + "' has not been initialised");
		}
		return init.getName();
	}

	@Override
	public List<String> getOperationNames() {
		List<String> names = new ArrayList<>();
		names.add(sClass.getSimpleName() + " -> " + tClass.getSimpleName());
		names.addAll(operations.stream().map(o -> o.name()).collect(Collectors.toList()));
		return names;
	}

	@Override
	public String toString() {
		if (getOperationNames().isEmpty()) {
			return "[name=" + name + ", input=" + init 
			+ "]";
		} else {
			return "[name=" + name + ", input=" + init + ", operationNames=" + getOperationNames()
				+ "]";
		}
	}

	@Override
	public SymbolicTensor<T> detach(String tensorName) {
		SymbolicTensor<T> detached = new MappedSymbolicTensorImpl<>(tensorName, init.detach(init.getName() + "Detached"), mapper, sClass, tClass);
		for (Operation<T> op : operations) {
			detached.performInlineOperation(op);
		}
		return detached;
	}

	@Override
	public Supplier<T> getInput() {
		if (init == null) {
			throw new IllegalStateException("Tensor '" + getName() + "' has not been initialised");
		}
		return () -> mapper.apply(init.evaluate());
	}

	@Override
	public int[] dimensions() {
		return dimensions;
	}

	@Override
	public List<String> dimensionNames() {
		return dimensionNames;
	}
	
	
}
