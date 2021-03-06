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


import java.util.function.Supplier;

import org.jvmpy.symbolictensors.TensorDataContainer;

public interface TensorOperations<T> extends Supplier<T>, TensorDataContainer, TensorOperationsMinimal<T> {

	T mul(float value);

	T add(float value);
	
	T sub(float value);


	T mul(T other);
	
	T div(T other);
	
	T sub(T other);

	int numel();
	
	T sum();

	T add(T other);

	T mean();
	
	T norm();

	T mul_(T other);
	
	T columnSums();
	
	T rowSums();

	T cloneTensor();


	T sub_(T other);

	T add_(T other);

	T matmul(T other);

	T t();

	Size size();
	
	T size_(Size size);
	
	
	T view(Size size);
	
	void close();
	


}