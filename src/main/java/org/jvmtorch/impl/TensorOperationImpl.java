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
package org.jvmtorch.impl;

import static org.jvmpy.python.Python.tuple;

import java.util.List;
import java.util.function.UnaryOperator;

import org.jvmpy.python.Tuple;
import org.jvmpy.symbolictensors.OperationImpl;
import org.jvmpy.symbolictensors.TensorDimensionsContainer;
import org.jvmtorch.torch.Size;
import org.jvmtorch.torch.TensorOperation;
import org.jvmtorch.torch.Torch;

public class TensorOperationImpl<T> extends OperationImpl<T> implements TensorOperation<T> {

	private UnaryOperator<Size> targetSizeMapping;
	
	public TensorOperationImpl(Torch torch, String name, UnaryOperator<T> operation, UnaryOperator<Size> targetSizeMapping) {
		super(name, operation, d -> dimensions(targetSizeMapping.apply(torch.Size(d.dimensions()).names_(names(d.dimensionNames())))));
		this.targetSizeMapping = targetSizeMapping;
	}

	
	private static Tuple<String> names(List<String> names) {
		if (names == null) {
			return null;
		} else {
			return tuple(names);
		}
	}
	
	@Override
	public UnaryOperator<Size> sizeMapping(Torch torch) {
		return targetSizeMapping;
	}


	private static TensorDimensionsContainer dimensions(Size size) {
		return new TensorDimensionsContainer() {

			@Override
			public int[] dimensions() {
				return size.dimensions();
			}

			@Override
			public List<String> dimensionNames() {

				Tuple<String> names = size.dimensionNames();
				if (names == null) {
					return null;
				} else {
					return names.asList();
				}
			}
			
		};
	}
}
