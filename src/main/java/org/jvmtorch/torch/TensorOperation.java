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

import static org.jvmpy.python.Python.tuple;

import java.util.List;
import java.util.function.UnaryOperator;

import org.jvmpy.symbolictensors.Operation;
import org.jvmpy.symbolictensors.TensorDimensionsContainer;

public interface TensorOperation<T> extends Operation<T> {

	default UnaryOperator<Size> sizeMapping(Torch torch) {
		return s-> mapSize(torch, s);
	}
	
	private Size mapSize(Torch torch, Size s) {
		TensorDimensionsContainer b =  dimensionsMapping().apply(new TensorDimensionsContainer() {

			@Override
			public int[] dimensions() {
				return s.dimensions();
			}

			@Override
			public List<String> dimensionNames() {
				return s.dimensionNames().asList();
			}});
		
		return torch.Size(b.dimensions()).names_(tuple(b.dimensionNames()));
	}
}
