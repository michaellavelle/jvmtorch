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

import java.util.List;
import java.util.function.Supplier;

/**
 * A SymbolicTensor allows an underlying TensorDataContainer of type T to be
 * manipulated in mathematical expressions, and for gradients
 * to be calculated.
 * 
 * An InitialSymbolicTensor is a tensor that is created in its own right -
 * not formed from a manipulation of an existing SymbolicTensor 
 * 
 * @author Michael Lavelle
 * 
 * @param <T> The type of TensorDataContainer to wrap inside the InitialSymbolicTensor
 */
public interface InitialSymbolicTensor<T extends TensorDataContainer> extends SymbolicTensor<T> {

	/**
	 * Initialises this InitialSymbolicTensor
	 * 
	 * @param inputName
	 * @param init
	 * @param dimensions
	 * @param dimensionNames
	 */
	void init(String inputName, Supplier<T> init, int[] dimensions, List<String> dimensionNames);
}
