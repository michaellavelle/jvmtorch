package org.jvmtorch.impl;

import org.jvmtorch.nn.Parameter;
import org.jvmtorch.nn.modules.MSELoss;
import org.jvmtorch.torch.Tensor;
import org.jvmtorch.torch.TensorOperations;
import org.jvmpy.python.OrderedDict;


public abstract class MSELossImpl<T extends TensorOperations<T>> implements MSELoss<T> {

	protected OrderedDict<Parameter<T>> parameters;

	@Override
	public OrderedDict<Parameter<T>> parameters() {
		return parameters;
	}

	@Override
	public void zero_grad() {
		if (parameters != null) {
			parameters.forEach(p -> p.getRight().zero_grad());
		}
	}

	@Override
	public Tensor<T> apply(Tensor<T> t, Tensor<T> u) {
		return forward(this, t, u);
	}
}
