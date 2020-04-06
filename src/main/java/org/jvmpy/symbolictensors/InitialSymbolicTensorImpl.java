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
import java.util.function.Supplier;
import java.util.stream.Collectors;

public class InitialSymbolicTensorImpl<T extends TensorDataContainer> implements InitialSymbolicTensor<T> {

	private String name;
	private Supplier<T> init;
	private String inputName;
	private List<Operation<T>> operations;
	private int[] dimensions;

	protected InitialSymbolicTensorImpl(String name) {
		this.name = name;
		this.operations = new ArrayList<>();
	}
	
	public InitialSymbolicTensorImpl(String name, String inputName, Supplier<T> init, int[] dimensions) {
		this(name);
		init(inputName, init, dimensions);
	}

	@Override
	public void init(String inputName, Supplier<T> init, int[] dimensions) {
		this.inputName = inputName;
		this.init = init;
		this.dimensions = dimensions;
	}
	
	@Override
	public T evaluate() {
		if (init == null) {
			throw new IllegalStateException("Tensor '" + getName() + "' has not been initialised");
		}
		T evaluation = init.get();
		for (Operation<T> operation : operations) {
			evaluation = operation.apply(evaluation);
		}
		return evaluation;
	}

	@Override
	public void performInlineOperation(Operation<T> operation) {
		this.operations.add(operation);
		this.dimensions = operation.dimensions();
		evaluate();
	}

	@Override
	public List<String> getAllOperationNames() {
		List<String> operationNames = new ArrayList<>();
		operationNames.addAll(getOperationNames());
		return operationNames;
	}

	@Override
	public SymbolicTensor<T> performUnaryMappingOperation(String newTensorName, Operation<T> operation) {
		SymbolicTensor<T> mappedTensor = new UnaryMappedSymbolicTensorImpl<>(newTensorName, this);
		mappedTensor.performInlineOperation(operation);
		return mappedTensor;
	}

	@Override
	public String getName() {
		return name;
	}

	@Override
	public String getInputName() {
		return inputName;
	}

	@Override
	public List<String> getOperationNames() {
		return operations.stream().map(o -> o.name()).collect(Collectors.toList());
	}

	@Override
	public String toString() {
		if (getOperationNames().isEmpty()) {
			return "[name=" + name + ", inputName=" + getInputName() 
			+ "]";
		} else {
			return "[name=" + name + ", inputName=" + getInputName() + ", operationNames=" + getOperationNames()
				+ "]";
		}
	}

	@Override
	public SymbolicTensor<T> detach(String tensorName) {
		return new InitialSymbolicTensorImpl<>(tensorName, inputName, init, dimensions);
	}

	@Override
	public Supplier<T> getInput() {
		return init;
	}

	@Override
	public List<Operation<T>> getOperations() {
		return this.operations;
	}

	@Override
	public int[] getDimensions() {
		return dimensions;
	}
	
}
