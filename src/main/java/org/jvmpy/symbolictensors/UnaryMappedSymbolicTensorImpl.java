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
import java.util.Collections;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;

public class UnaryMappedSymbolicTensorImpl<T extends TensorDataContainer> implements UnaryMappedSymbolicTensor<T> {

	private String name;
	protected SymbolicTensor<T> init;
	private List<Operation<T>> operations;
	private int[] dimensions;

	protected UnaryMappedSymbolicTensorImpl(String name) {
		this.name = name;
		this.operations = new ArrayList<>();
	}
	
	public UnaryMappedSymbolicTensorImpl(String name, SymbolicTensor<T> init) {
		this(name);
		init(init);
	}

	@Override
	public void init(SymbolicTensor<T> init) {
		this.init = init;
		this.dimensions = init.getDimensions();
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
		this.dimensions = operation.dimensions();
	}

	@Override
	public SymbolicTensor<T> performUnaryMappingOperation(String newTensorName, Operation<T> operation) {
		SymbolicTensor<T> mappedTensor = new UnaryMappedSymbolicTensorImpl<>(newTensorName, this);
		mappedTensor.performInlineOperation( operation);
		evaluate();
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
		names.addAll(operations.stream().map(o -> o.name()).collect(Collectors.toList()));
		return names;
	}

	@Override
	public List<String> getAllOperationNames() {
		List<String> operationNames = new ArrayList<>();
		operationNames.addAll(init.getAllOperationNames());
		operationNames.addAll(getOperationNames());
		return operationNames;
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
		
		Supplier<T> input = getInput();
		String inputName = getInputName();
		List<Operation<T>> operations = new ArrayList<>();
		operations.addAll(this.operations);
		Collections.reverse(operations);
		while (input instanceof SymbolicTensor) {
			SymbolicTensor<T> tensor = (SymbolicTensor<T>)input;
			List<Operation<T>> ops = new ArrayList<>();
			ops.addAll(tensor.getOperations());
			Collections.reverse(ops);
			operations.addAll(ops);
			input = tensor.getInput();
			inputName = tensor.getInputName();
		}
		Collections.reverse(operations);
		
		SymbolicTensor<T> detached = new InitialSymbolicTensorImpl<>(tensorName, inputName, input, dimensions);
		for (Operation<T> op : operations) {
			detached.performInlineOperation(op);
		}
		return detached;
	}

	@Override
	public SymbolicTensor<T> getInput() {
		if (init == null) {
			throw new IllegalStateException("Tensor '" + getName() + "' has not been initialised");
		}
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
