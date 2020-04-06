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
package org.jvmtorch;

import org.jvmtorch.nn.Functional;
import org.jvmtorch.nn.NN;
import org.jvmtorch.torch.TensorOperations;
import org.jvmtorch.torch.Torch;
import org.jvmtorch.torch.optim.Optim;

/**
 * A Java-style factory for creation of implementations of
 * Torch, Functional, NN and Optim for a specific type of
 * TensorOperations.
 * 
 * Python-style access to these implementations is provided
 * via JvmTorch attributes of torch, nn, F and optim.
 * 
 * @author Michael Lavelle
 *
 * @param <T> The type of TensorOperations required by this
 * factory.
 */
public interface JvmTorchFactory<T extends TensorOperations<T>> {
	
	Torch<T> createTorch();
	
	Functional<T> createFunctional();
	
	NN<T> createNN();
	
	Optim<T> createOptim();
	
}
