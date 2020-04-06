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

import org.jvmtorch.nn.BaseModule;
import org.jvmtorch.torch.Tensor;
import org.jvmtorch.torch.TensorOperations;

import java.util.function.BinaryOperator;

public interface Loss<T extends TensorOperations<T>, L extends Loss<T, L>> extends BaseModule<L, T>, BinaryOperator<Tensor<T>> {

	Tensor<T> forward(L self, Tensor<T> input, Tensor<T> target);

}
