package org.jvmtorch.impl;

import org.jvmtorch.nn.Parameter;
import org.jvmtorch.torch.GradFunction;
import org.jvmtorch.torch.Tensor;
import org.jvmtorch.torch.TensorOperation;
import org.jvmtorch.torch.TensorOperations;
import org.jvmpy.python.Tuple;

import java.util.List;

import static org.jvmpy.python.Python.True;

public class ParameterImpl<T extends TensorOperations<T>> extends TensorAdapter<T> implements Parameter<T> {

	public ParameterImpl(Tensor<T> adapted) {
		super(adapted.requires_grad_(True));
	}

	@Override
	public void zero_grad() {
		grad_(null);
	}

	@Override
	public String toString() {
		//return "Parameter containing:\n" + this.adapted.toString();
		return "Parameter";
	}

	@Override
	public Parameter<T> sub_(Tensor<T> other) {
		super.sub_(other);
		return this;
	}

	@Override
	public Parameter<T> mul_(Tensor<T> other) {
		super.mul_(other);
		return this;
	}

	@Override
	public Parameter<T> withNextFunctions(String name, List<TensorOperation<Tensor<T>>> tensorOperations, Tuple<Tuple<GradFunction<T>>> nextFunctions) {
		super.withNextFunctions(name, tensorOperations, nextFunctions);
		return this;
	}

	@Override
	public Parameter<T> requires_grad_(boolean requires_grad) {
		super.requires_grad_(requires_grad);
		return this;
	}

	@Override
	public Parameter<T> grad_(Tensor<T> grad) {
		super.grad_(grad);
		return this;
	}

	@Override
	public Parameter<T> add_(Tensor<T> other) {
		super.add_(other);
		return this;
	}
}
