package org.jvmtorch.impl;

import org.jvmtorch.torch.*;
import org.jvmpy.python.PythonClass;
import org.jvmpy.python.Tuple;
import org.jvmpy.symbolictensors.InitialSymbolicTensorImpl;
import org.jvmpy.symbolictensors.OperationImpl;
import org.jvmpy.symbolictensors.SymbolicTensor;
import org.jvmtorch.torch.*;

import java.util.Arrays;
import java.util.List;
import java.util.function.UnaryOperator;

import static org.jvmpy.python.Python.True;
import static org.jvmpy.python.Python.tuple;


public abstract class TensorBase<T extends TensorOperations<T>> extends PythonClass<Tensor<T>> implements Tensor<T> {

	protected SymbolicTensor<T> symbolicTensor;
	protected Torch<T> torch;
	protected Tensor<T> grad;
	protected GradFunction<T> grad_fn;
	protected boolean requires_grad;

	public TensorBase(Torch<T> torch, SymbolicTensor<T> symbolicTensor) {
		this.torch = torch;
		this.symbolicTensor = symbolicTensor;
	}

	public TensorBase(Torch<T> torch, String name, String inputName, T tensorOperations) {
		this.torch = torch;
		this.symbolicTensor = toSymbolicTensor(name, inputName, tensorOperations);
	}

	@Override
	public float[] getDataAsFloatArray() {
		float[] data =  this.symbolicTensor.evaluate().data();
		if (data.length != this.numel()) {
			throw new IllegalStateException();
		}
		return data;
	}

	protected abstract Tensor<T> createDefaultTensor(Torch<T> torch, SymbolicTensor<T> tensor);


		@Override
	public Size size() {
		if (symbolicTensor.getDimensions().length > 1) {
			int[] dims = new int[symbolicTensor.getDimensions().length - 1];
			for (int i = 1; i < symbolicTensor.getDimensions().length; i++) {
				dims[i - 1] = symbolicTensor.getDimensions()[i];
			}
			return new Size(symbolicTensor.getDimensions()[0], dims);
		} else {
			return new Size(symbolicTensor.getDimensions()[0]);
		}
	}

	@Override
	public Tensor<T> mul(float value) {
		return toTensor(symbolicTensor
				.performUnaryMappingOperation("Mul", new OperationImpl<>("Mul", t -> t.mul(value), size().getDimensions())), "MulBackward", new TensorOperationImpl<>("MulBackward", g -> g.mul(value), size()));
	}

	@Override
	public Tensor<T> add(float value) {
		return toTensor(symbolicTensor
				.performUnaryMappingOperation("Add", new OperationImpl<>("Add", t -> t.add(value), size().getDimensions())), "AddBackward", new TensorOperationImpl<Tensor<T>>("AddBackward", g -> g, size()));
	}

	@Override
	public Tensor<T> mean() {
		// TODO backward
		return toTensor(symbolicTensor
				.performUnaryMappingOperation("Mean", new OperationImpl<>("Mean", t -> t.mean(), new int[]{1, 1})), "MeanBackward", new TensorOperationImpl<>("MeanBackward", g -> g.mul(1f / numel()), 1, 1));
	}

	@Override
	public Tensor<T> mul(Tensor<T> other) {
		if (other.requires_grad()) {
			return toTensor(symbolicTensor
					.performUnaryMappingOperation("Mul", new OperationImpl<>("Mul", t -> t.mul(fromTensor(other)), size().getDimensions())), "MulBackward", Arrays.asList(this, other), Arrays.asList(new TensorOperationImpl<>("MulBackward", g -> g.mul(other), size()), new TensorOperationImpl<>("MulBackward", g -> g.mul(this), other.size())));
		} else {
			return toTensor(symbolicTensor
					.performUnaryMappingOperation("Mul", new OperationImpl<>("Mul", t -> t.mul(fromTensor(other)), size().getDimensions())), "MulBackward", new TensorOperationImpl<>("MulBackward", g -> g.mul(other), size()));
		}
	}


	@Override
	public int numel() {

		int numel = 1;
		for (int dim : symbolicTensor.getDimensions()) {
			numel = numel * dim;
		}
		return numel;

	}

	@Override
	public Tensor<T> add(Tensor<T> other) {
		if (other.requires_grad()) {
			return toTensor(symbolicTensor
					.performUnaryMappingOperation("Add", new OperationImpl<>("Add", t -> t.add(fromTensor(other)), size().getDimensions())),"AddBackward",  Arrays.asList(this, other),   Arrays.asList(new TensorOperationImpl<>("AddBackward", g -> g, size()), new TensorOperationImpl<>("AddBackward", g -> g, other.size())));
		} else {
			return toTensor(symbolicTensor
					.performUnaryMappingOperation("Add", new OperationImpl<>("Add", t -> t.add(fromTensor(other)), size().getDimensions())), "AddBackward", new TensorOperationImpl<>("AddBackward", g -> g, size()));
		}

	}

	protected Tensor<T> toTensor(SymbolicTensor<T> tensor, String name, TensorOperation<Tensor<T>> operation) {
		if (this.requires_grad()) {
			//System.out.println("GRAD:" + this.grad_fn());
			if (this.grad_fn() != null) {
				Tensor<T> t = createDefaultTensor(torch, tensor)
						.withNextFunctions(name, Arrays.asList(operation),
								tuple(tuple(this.grad_fn()))
						);

				return t.requires_grad_(True);
			} else {
				int firstDim = tensor.getDimensions()[0];
				int[] otherDims = new int[tensor.getDimensions().length - 1];
				for (int i = 0; i < otherDims.length; i++) {
					otherDims[i] = tensor.getDimensions()[i+1];
				}
				Tensor<T> t = createDefaultTensor(torch, tensor)
						.withNextFunctions(name, Arrays.asList(operation),
								tuple(tuple(new GradFunctionImpl<T>("AccumulateGrad", Arrays.asList(new TensorOperationImpl<>("AccumulateGrad", l -> l, firstDim, otherDims)),
										this, null)))
						);

				return t.requires_grad_(True);
			}

		} else {
			return createDefaultTensor(torch, tensor);
		}
	}


	private GradFunction<T> getGradFunction(Tensor<T> tensor) {
		if (tensor.grad_fn() != null) {
			return tensor.grad_fn();
		} else {
			return new GradFunctionImpl<T>("AccumulateGrad", Arrays.asList(new TensorOperationImpl<>("AccumulateGrad", l -> l, tensor.size())),
					tensor, null);
		}
	}

	protected Tensor<T> toTensor(SymbolicTensor<T> tensor, String name, List<Tensor<T>> tensors, List<TensorOperation<Tensor<T>>> operations) {
		if (this.requires_grad()) {
			if (operations.size() == 2) {
				Tensor<T> t = createDefaultTensor(torch, tensor)
						.withNextFunctions(name, operations,
								tuple(tuple(getGradFunction(tensors.get(0))), tuple(getGradFunction(tensors.get(1)))));

				return t.requires_grad_(True);

			} else if (operations.size() == 1){
				Tensor<T> t = createDefaultTensor(torch, tensor)
						.withNextFunctions(name, operations,
								tuple(tuple(getGradFunction(this)))
						);

				return t.requires_grad_(True);
			} else {
				throw new RuntimeException("Operations size:" + operations.size());
			}
		} else {
			return createDefaultTensor(torch, tensor);
		}
	}

	protected abstract T fromTensor(Tensor<T> other);
	
	@Override
	public T toTensorOperations() {
		return fromTensor(this);
	}

	@Override
	public float[] data() {
		return this.getDataAsFloatArray();
	}

	@Override
	public Tensor<T> performUnaryMappingOperation(String newTensorName, TensorOperation<T> operation, TensorOperation<Tensor<T>> backwardOp) {
		return toTensor(symbolicTensor
				.performUnaryMappingOperation(newTensorName, new OperationImpl<>(operation.name(), t -> operation.apply(t), operation.dimensions())),backwardOp.name(), backwardOp);
	}

	@Override
	public Tensor<T> sub_(Tensor<T> other) {
		symbolicTensor
		.performInlineOperation(new OperationImpl<>("Sub", t -> { t.sub_(fromTensor(other)); return t; }, size().getDimensions()));
		return this;
	}

	@Override
	public Tensor<T> mul_(Tensor<T> other) {
		symbolicTensor
				.performInlineOperation(new OperationImpl<>("Mul", t -> { t.mul_(fromTensor(other)); return t; }, size().getDimensions()));
		return this;
	}

	@Override
	public Tensor<T> add_(Tensor<T> other) {
		symbolicTensor
		.performInlineOperation(new OperationImpl<>("Add", t -> { t.add_(fromTensor(other)); return t; }, size().getDimensions()));
		return this;
	}



	@Override
	public Tensor<T> matmul(Tensor<T> other) {
		if (other.requires_grad()) {
			return toTensor(symbolicTensor
							.performUnaryMappingOperation("Matmul", new OperationImpl<>("Matmul", t -> t.matmul(fromTensor(other)), new int[] { this.size().get(0), other.size().get(1)})),
					"MatmulBackward", Arrays.asList(this, other), Arrays.asList(new TensorOperationImpl<>("MatmulBackward", g -> g.matmul(other.transpose()), size()), new TensorOperationImpl<>("MatmulBackward", g -> g.transpose().matmul(this).transpose(), other.size())));
		} else {
			return toTensor(symbolicTensor
							.performUnaryMappingOperation("Matmul", new OperationImpl<>("Matmul", t -> t.matmul(fromTensor(other)), new int[] { this.size().get(0), other.size().get(1)})),
					"MatmulBackward", new TensorOperationImpl<Tensor<T>>("MatmulBackward", g -> g.matmul(other.transpose()), size()));
		}

	}

	@Override
	public Tensor<T> transpose() {
		if (size().get(0) == 0 || size().get(1) == 0) {
			throw new IllegalStateException();
		}
		return toTensor(symbolicTensor
				.performUnaryMappingOperation("Transpose", new OperationImpl<>("Transpose",t -> t.transpose(), new int[] {size().get(1), size().get(0)})), "TransposeBackward", new TensorOperationImpl<>("TransposeBackward", t -> t.transpose(), this.size()));
	}

	private SymbolicTensor<T> toSymbolicTensor(String name, String inputName, T tensor) {
		Integer first = tensor.size().get(0);
		Integer second = tensor.size().get(1);

		return new InitialSymbolicTensorImpl<T>(name, inputName, () -> tensor, new int[] {first, second});
	}


	@Override
	public Tensor<T> t() {
		return transpose();
	}

	@Override
	public void backward(Tensor<T> gradient) {
		if (gradient == null) {
			if (this.numel() == 1) {
				gradient = torch.ones(1, 1);
			} else {
				throw new RuntimeException("grad can be implicitly created only for scalar outputs");
			}
		}

		backward2(this.grad_fn(), gradient);

	}


	public static <T extends TensorOperations<T>> void backward2(GradFunction<T> gf, Tensor<T> back) {

		if (back == null) {
			throw new RuntimeException("Back is null");
		}


		if (gf != null) {
			if (gf.next_functions() != null && gf.next_functions().getComponents().length > 0) {
				int index = 0;

				for (UnaryOperator<Tensor<T>> operation : gf.operations()) {
					Tensor<T> back2 = operation.apply(back);
					Tuple<GradFunction<T>> gf2Tuple = gf.next_functions().get(index);
					GradFunction<T> gf2 = gf2Tuple.get(0);
					backward2(gf2, back2);
					index++;
				}
			} else {
				Tensor<T> target = gf.variable();
				if (target != null) {
					if (target.grad() == null) {
						target.grad_(back);
					} else {
						target.grad().add_(back);
					}
					target.backward(back);
				}}
		}
	}

	@Override
	public Tensor<T> grad_fn_(GradFunction<T> grad_fn) {
		this.grad_fn = grad_fn;
		return this;
	}

	@Override
	public boolean requires_grad() {
		return requires_grad;
	}

	@Override
	public Tensor<T> grad_(Tensor<T> grad) {
		this.grad = grad;
		return this;
	}

	@Override
	public void backward() {
		backward(null);
	}

	@Override
	public Tensor<T> grad() {
		return grad;
	}
	
	@Override
	public GradFunction grad_fn() {
		return grad_fn;
	}

	@Override
	public Tensor<T> requires_grad_(boolean requires_grad) {
		this.requires_grad = true;
		return this;
	}

	@Override
	public Tensor<T> withNextFunctions(String name, List<TensorOperation<Tensor<T>>> operations, Tuple<Tuple<GradFunction<T>>> nextFunctions) {
		if (self.grad_fn()  != null) {
			throw new RuntimeException("Already set");
		}
		this.grad_fn = new GradFunctionImpl<T>(name,  operations, null, nextFunctions);
		return this;
	}

	@Override
	public Tensor<T> get() {
		return this;
	}
}
