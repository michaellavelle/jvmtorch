package org.jvmtorch.impl.operations.tensortensor;

import org.jvmtorch.torch.TensorOperations;
import org.jvmtorch.torch.Torch;

public abstract class DifferentiableTensorTensorFunctionBase<T extends TensorOperations<T>> implements DifferentiableTensorTensorFunction<T> {

	protected Torch torch;
	
	public DifferentiableTensorTensorFunctionBase(Torch torch) {
		this.torch = torch;
	}
}
