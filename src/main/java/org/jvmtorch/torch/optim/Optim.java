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
package org.jvmtorch.torch.optim;

import org.jvmtorch.nn.Parameter;
import org.jvmtorch.torch.TensorOperations;
import org.jvmpy.python.OrderedDict;

/**
 * Factory for different types of Optimiser, with factory method
 * signatures matching the Optimiser constructors - enabling construction
 * without requiring the "new" keyword for more python-like syntax.
 * 
 * @author Michael Lavelle
 *
 */
public interface Optim<T extends TensorOperations<T>> {
		
	/**
	 * Create a SGD (Standard Gradient Descent) Optimiser
	 * 
	 * @param parameters The parameters to optimise
	 * @param learningRate The learning rate.
	 * @return a SGD (Standard Gradient Descent) Optimiser
	 */
	Optimiser SGD(OrderedDict<Parameter<T>> parameters, Number learningRate);
}
