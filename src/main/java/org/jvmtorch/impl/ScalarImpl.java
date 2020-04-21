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

import org.jvmpy.symbolictensors.SymbolicTensor;
import org.jvmtorch.torch.Size;
import org.jvmtorch.torch.Tensor;
import org.jvmtorch.torch.TensorData;
import org.jvmtorch.torch.TensorDataAdapter;
import org.jvmtorch.torch.TensorDataConverter;
import org.jvmtorch.torch.Torch;

public class ScalarImpl extends TensorBase {
	
	public ScalarImpl(Torch torch, String name, String inputName,
			TensorData tensorData) {
		super(torch, new ScalarDataConverter(torch), tensorData);
	}

	public ScalarImpl(Torch torch, TensorDataConverter<?> tensorDataConverter,
			SymbolicTensor<TensorData, Size> symbolicTensor) {
		super(torch, new ScalarDataConverter(torch), symbolicTensor);
	}
	
	public ScalarImpl(Torch torch, String name, String inputName,
			float value) {
		super(torch, new ScalarDataConverter(torch), new TensorDataAdapter<>(new ScalarOperations(torch, value), new ScalarDataConverter(torch)));
	}

	@Override
	protected Tensor createDefaultTensor(Torch torch, SymbolicTensor<TensorData, Size> tensor) {
		if (tensor.size().dimensions().length == 0) { 
			return new ScalarImpl(torch, new ScalarDataConverter(torch), tensor);
		} else {
			return torch.tensor(tensor.get());
		}
	}
}
