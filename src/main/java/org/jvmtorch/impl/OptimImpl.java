package org.jvmtorch.impl;

import org.jvmtorch.nn.Parameter;
import org.jvmtorch.torch.TensorOperations;
import org.jvmtorch.torch.optim.Optim;
import org.jvmpy.python.OrderedDict;

public class OptimImpl<T extends TensorOperations<T>> implements Optim<T> {
		
	public SGD<T> SGD(OrderedDict<Parameter<T>> parameters, Number learningRate)  {
		return new SGD<>(parameters, learningRate);
	}
}
