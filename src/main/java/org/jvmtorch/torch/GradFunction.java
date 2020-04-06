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
package org.jvmtorch.torch;

import org.jvmpy.python.Tuple;

import java.util.List;


public abstract class GradFunction<T extends TensorOperations<T>> {

	private String name;
	private List<TensorOperation<Tensor<T>>> operations;
	private Tuple<Tuple<GradFunction<T>>> next_functions;
	
	public Tuple<Tuple<GradFunction<T>>> next_functions() {
		return next_functions;
	}
		
	public GradFunction(String name, List<TensorOperation<Tensor<T>>> operations, Tuple<Tuple<GradFunction<T>>> next_functions) {
		this.name = name;
		this.next_functions = next_functions;
		this.operations = operations;
	}
	
	public String name() {
		return name;
	}
	
	public abstract Tensor<T> variable();
	
	public List<TensorOperation<Tensor<T>>> operations() {
		return operations;
	}
}
