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

import java.util.List;

import org.jvmpy.python.Tuple;

/**
 * Encapsulates a JVMTorch GradFunction.
 * 
 * @author Michael Lavelle
 *
 */
public interface GradFunction {

	/**
	 * @return A list of TensorOperation within this grad
	 * function.
	 */
	List<TensorOperation<Tensor>> operations();
	
	/**
	 * @return The name of this grad function.
	 */
	String name();
	
	/**
	 * @return A tuple of GradFunctionss within the
	 * computation graph.
	 */
	Tuple<Tuple<GradFunction>> next_functions();
	
	/**
	 * @return The target variable
	 */
	Tensor variable();

}
