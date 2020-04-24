package org.jvmtorch.impl;

import org.jvmtorch.torch.Tensor;
import org.jvmtorch.torch.TensorDataConverter;
import org.jvmtorch.torch.TensorOperations;
import org.jvmtorch.torch.Torch;

public abstract class TorchImpl<T extends TensorOperations<T>> implements Torch {

	public static int LONG = 1;

	protected TensorDataConverter<T> tensorDataConverter;
	
	public TorchImpl(TensorDataConverter<T> tensorDataConverter) {
		this.tensorDataConverter = tensorDataConverter;
	}

	@Override
	public Tensor tensor(float value) {
		return new ScalarImpl(this, value);
	}

	@Override
	public Tensor add(Tensor first, Tensor second) {
		return first.add(second);
	}

	@Override
	public Tensor add(Tensor first, float value) {
		return first.add(value);
	}

	@Override
	public Tensor mul(Tensor first, Tensor second) {
		return first.mul(second);
	}

	@Override
	public Tensor mul(Tensor first, float value) {
		return first.mul(value);
	}

	@Override
	public Tensor randn(int... dimensions) {
		return randn(Size(dimensions));
	}

	@Override
	public Tensor rand(int... dimensions) {
		return rand(Size(dimensions));
	}

	@Override
	public Tensor empty(int... dimensions) {
		return empty(Size(dimensions));
	}

	@Override
	public Tensor zeros(int... dimensions) {
		return zeros(Size(dimensions));
	}

	@Override
	public Tensor ones(int... dimensions) {
		return ones(Size(dimensions));
	}

	@Override
	public Tensor tensor(float[] data, int... dimensions) {
		return tensor(data, dimensions.length == 0 ? Size(data.length) : Size(dimensions));
	}

}
