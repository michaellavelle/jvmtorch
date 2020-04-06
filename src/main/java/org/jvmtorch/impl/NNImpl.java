package org.jvmtorch.impl;

import org.jvmtorch.nn.Functional;
import org.jvmtorch.nn.NN;
import org.jvmtorch.nn.Parameter;
import org.jvmtorch.torch.Tensor;
import org.jvmtorch.torch.TensorOperations;
import org.jvmtorch.torch.Torch;


public abstract class NNImpl<T extends TensorOperations<T>> implements NN<T> {

	protected Torch<T> torch;
	protected Functional<T> f;

	public NNImpl(Torch<T> torch, Functional<T> f) {
		this.torch = torch;
		this.f = f;
	}

	public Torch<T> torch() {
		return torch;
	}
	
	public Functional<T> f() {
		return f;
	}

	public Parameter<T> Parameter(Tensor<T> tensor) {
		return new ParameterImpl<>(tensor);
	}

	@Override
	public Parameter<T> Parameter(int...dims) {
		for (int i : dims) {
			if (i == 0) {
				throw new IllegalArgumentException();
			}
		}
		return new ParameterImpl<>(torch.randn(dims).mul(0.01f));
	}
}
