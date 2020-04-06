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

public interface Torch<T extends TensorOperations<T>> {

	public static int LONG = 1;
	
	Tensor<T> empty(int i, int j);
	Tensor<T> zeros(int i, int j);
	Tensor<T> ones(int i, int j);
	Tensor<T> randn(int i, int j, int k, int l);
	Tensor<T> randn(int i, int j);
	Tensor<T> rand(int i, int j);
	Tensor<T> randn(int i);
	Tensor<T> randn(int... sizes);
}
