package org.jvmtorch.impl;

import org.jvmtorch.nn.Parameter;
import org.jvmtorch.torch.TensorOperations;
import org.jvmtorch.torch.optim.Optimiser;
import org.jvmpy.python.OrderedDict;

public class SGD<T extends TensorOperations<T>> extends OptimiserImpl<SGD> implements Optimiser {

	protected Float learning_rate;

	public SGD(OrderedDict<Parameter<T>> parameters, Number learning_rate) {
		this.learning_rate = learning_rate == null ? null : learning_rate.floatValue();
	}


	@Override
	public SGD self() {
		return this;
	}

	@Override
	public void step() {

	}

	@Override
	public void zero_grad() {

	}

	@Override
	public String toString() {
		return "SGD [lr=" + learning_rate + "]";
	}
}
