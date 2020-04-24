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

import static org.jvmpy.python.Python.True;
import static org.jvmpy.python.Python.tuple;

import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.jvmpy.python.PythonClass;
import org.jvmpy.python.Tuple;
import org.jvmpy.symbolictensors.Operation;
import org.jvmpy.symbolictensors.SymbolicTensor;
import org.jvmpy.symbolictensors.SymbolicTensorAdapter;
import org.jvmtorch.impl.operations.tensorscalar.DifferentiableTensorScalarFunction;
import org.jvmtorch.impl.operations.tensorscalar.TensorScalarImpl;
import org.jvmtorch.impl.operations.tensortensor.DifferentiableTensorTensorFunction;
import org.jvmtorch.impl.operations.tensortensor.TensorAddition;
import org.jvmtorch.torch.GradFunction;
import org.jvmtorch.torch.Size;
import org.jvmtorch.torch.SizeMatcher;
import org.jvmtorch.torch.Tensor;
import org.jvmtorch.torch.TensorData;
import org.jvmtorch.torch.TensorDataConverter;
import org.jvmtorch.torch.TensorOperation;
import org.jvmtorch.torch.TensorOperations;
import org.jvmtorch.torch.Torch;

public abstract class TensorBase extends PythonClass<Tensor> implements Tensor {

	public SymbolicTensor<TensorData, Size> symbolicTensor;
	protected Torch torch;
	protected Tensor grad;
	protected GradFunction grad_fn;
	protected boolean requires_grad;
	protected boolean create_graph;
	protected TensorDataConverter<?> tensorDataConverter;
	// TODO - make thread safe.
	protected Boolean pre_operation_requires_grad;

	public Torch torch() {
		return torch;
	}

	public TensorBase(Torch torch, TensorDataConverter<?> tensorDataConverter,
			SymbolicTensor<TensorData, Size> symbolicTensor) {
		this.torch = torch;
		this.symbolicTensor = symbolicTensor;
		this.tensorDataConverter = tensorDataConverter;
	}

	public TensorBase(Torch torch, TensorDataConverter<?> tensorDataConverter, TensorData tensorData) {
		this.torch = torch;
		this.symbolicTensor = toSymbolicTensor(tensorData);
		this.symbolicTensor.size_(tensorData.size());
		this.tensorDataConverter = tensorDataConverter;
	}

	@Override
	public String toString() {
		TensorData tensorData = symbolicTensor.evaluate();
		TensorOperations<?> tensorOperations = tensorDataConverter.createTensorOperationsFromTensorData(tensorData);
		String s = tensorOperations.toString();
		if (requires_grad() && grad_fn() == null) {
			if (names() != null && names().length() > 0) {
				return "tensor(" + s.toString() + ", requires_grad=True, names=" + names() + ")";
			} else {
				return "tensor(" + s.toString() + ", requires_grad=True)";
			}
		} else if (grad_fn() != null) {
			if (names() != null && names().length() > 0) {
				return "tensor(" + s.toString() + ", grad_fn=" + grad_fn().toString() + ", names=" + names() + ")";
			} else {
				return "tensor(" + s.toString() + ", grad_fn=" + grad_fn().toString() + ")";
			}
		} else {
			if (names() != null && names().length() > 0) {
				return "tensor(" + s.toString() + ", names=" + names() + ")";
			} else {
				return "tensor(" + s.toString() + ")";

			}
		}
	}

	/**
	 * Perform an inline operation on the underlying tensor, potentially lazily.
	 * 
	 * @param operation The operation to perform.
	 */
	void performInlineOperation(Operation<TensorData, Size> operation) {
		if (requires_grad) {
			// TODO. Amend this method so it throws exception unless in a backward pass.
			throw new IllegalStateException("a leaf Variable that requires grad has been used "
					+ "in an in-place operation:" + operation.name());
		}
		symbolicTensor.performInlineOperation(operation);
	}

	/**
	 * Perform an operation
	 * 
	 * @param newTensorName
	 * @param operation
	 * @return
	 */
	SymbolicTensor<TensorData, Size> performUnaryMappingOperation(Operation<TensorData, Size> operation) {
		return symbolicTensor.performUnaryMappingOperation(operation);
	}

	@Override
	public float[] getDataAsFloatArray() {
		float[] data = this.symbolicTensor.evaluate().getDataAsFloatArray();
		if (data.length != this.numel()) {
			throw new IllegalStateException();
		}
		return data;
	}

	public Tensor size_(Size size) {
		Size existingSize = size();
		if (existingSize.numel() != size.numel()) {
			throw new IllegalArgumentException(existingSize.numel() + ":" + size.numel());

		} else {
			this.symbolicTensor.size_(size);
		}
		return this;
	}

	@Override
	public Size size() {
		return symbolicTensor.size();
	}

	protected Tensor addScalarVector(Tensor other) {
		return applyBinaryTensorOperation(new TensorAddition<>(torch), other);
	}
	
	protected Tensor applyInlineOperation(Consumer<TensorData> consumer) {
		performInlineOperation(new TensorOperationImpl<TensorData, Size>(torch, "add_", 
				toUnaryOperator(consumer),
				s -> s));
		return this;
	}
	
	private UnaryOperator<TensorData> toUnaryOperator(Consumer<TensorData> consumer) {
		return t -> { consumer.accept(t); return t; };
	}
	
	public TensorOperation<Tensor, Size> createBackwardOperation(String operationName, UnaryOperator<Tensor> backPropFunction) {
		return new TensorOperationImpl<>(torch, operationName + "Backward",
				backPropFunction, s -> size());
	}
	
	public TensorOperation<TensorData, Size> createForwardOperation(String operationName, UnaryOperator<TensorData> forwardPropFunction, UnaryOperator<Size> sizeFunction) {
		return new TensorOperationImpl<>(torch, operationName,
				forwardPropFunction, sizeFunction);
	}
	
	protected Tensor applyUnaryTensorOperation(String operationName, 
			UnaryOperator<TensorData> forwardPropFunction, 
			UnaryOperator<Size> sizeFunction, 
			UnaryOperator<Tensor> backPropFunction) {
			return toTensor(
					performUnaryMappingOperation(
							createForwardOperation(operationName, forwardPropFunction, sizeFunction)),
						operationName + "Backward", 
							createBackwardOperation(operationName, backPropFunction));
	}

	protected Tensor applyBinaryTensorOperation(DifferentiableTensorTensorFunction<TensorData> tensorOperation, Tensor other) {

		UnaryOperator<TensorData> forwardPropFunction = tensorOperation.forwardPropFunction(fromTensor(other));

		Pair<UnaryOperator<Tensor>, UnaryOperator<Tensor>> backPropFunctions = tensorOperation
				.backPropFunctions(new ImmutablePair<>(this, other));

		UnaryOperator<Size> sizeFunction = tensorOperation.sizeFunction(new ImmutablePair<>(size(), other.size()));

		if (other.requires_grad()) {
			pre_operation_requires_grad = this.requires_grad;; 
			//boolean original = this.requires_grad;
			this.requires_grad_(true);
			Tensor ret = toTensor(
					performUnaryMappingOperation(
							createForwardOperation(tensorOperation.name(), forwardPropFunction, sizeFunction)),
							tensorOperation.name() + "Backward", Arrays.asList(this, other),
							Arrays.asList(
									createBackwardOperation(tensorOperation.name(), backPropFunctions.getLeft()),
									other.createBackwardOperation(tensorOperation.name(), backPropFunctions.getRight())
									)
							);
			
			this.requires_grad_(pre_operation_requires_grad);
			return ret;
		} else {
			return toTensor(
					performUnaryMappingOperation(
							createForwardOperation(tensorOperation.name(), forwardPropFunction, sizeFunction)),
							tensorOperation.name() + "Backward", 
							createBackwardOperation(tensorOperation.name(), backPropFunctions.getLeft()));
		}
	}

	protected Tensor applyScalarOperation(DifferentiableTensorScalarFunction<TensorData> tensorOperation, float scalar) {

		UnaryOperator<TensorData> forwardPropFunction = tensorOperation.forwardPropFunction(scalar);

		UnaryOperator<Tensor> backPropFunction = tensorOperation.backPropFunction(new TensorScalarImpl<>(this, scalar));

		return toTensor(
				performUnaryMappingOperation(
						createForwardOperation(tensorOperation.name(), forwardPropFunction, s -> s)),
				tensorOperation.name() + "Backward",
				createBackwardOperation(tensorOperation.name(), backPropFunction));

	}

	@Override
	public int numel() {
		return Math.max(Arrays.stream(symbolicTensor.size().dimensions()).reduce(1, (a, b) -> a * b), 1);
	}
	
	protected abstract Tensor createDefaultTensor(Torch torch, SymbolicTensor<TensorData, Size> tensor);

	protected Tensor toTensor(SymbolicTensor<TensorData, Size> tensor, String name,
			TensorOperation<Tensor, Size> operation) {
		if (this.requires_grad()) {
			if (this.grad_fn() != null) {
				Tensor t = createDefaultTensor(torch, tensor).withNextFunctions(name, Arrays.asList(operation),
						tuple(tuple(this.grad_fn())));
				return t.requires_grad_(True).create_graph_(create_graph);
			} else {

				if (tensor.size().dimensionNames() != null) {
					Tensor t = createDefaultTensor(torch, tensor).withNextFunctions(name, Arrays.asList(operation),
							tuple(tuple(new GradFunctionImpl("AccumulateGrad",
									Arrays.asList(new TensorOperationImpl<>(torch, "AccumulateGrad", l -> l, s -> s)),
									this, null))));

					return t.requires_grad_(True).create_graph_(create_graph);

				} else {
					Tensor t = createDefaultTensor(torch, tensor).withNextFunctions(name, Arrays.asList(operation),
							tuple(tuple(new GradFunctionImpl("AccumulateGrad",
									Arrays.asList(new TensorOperationImpl<>(torch, "AccumulateGrad", l -> l, s -> s)),
									this, null))));

					return t.requires_grad_(True).create_graph_(create_graph);

				}
			}

		} else {
			return createDefaultTensor(torch, tensor).create_graph_(create_graph);
		}
	}

	private GradFunction getGradFunction(Tensor tensor) {
		if (tensor.grad_fn() != null) {
			return tensor.grad_fn();
		} else {
			return new GradFunctionImpl("AccumulateGrad",
					Arrays.asList(new TensorOperationImpl<>(torch, "AccumulateGrad", l -> l, s -> s)), tensor, null);
		}
	}

	private Tuple<Tuple<GradFunction>> getGradFunctionTuples(List<Tensor> tensors) {
		return tuple(tensors.stream().map(t -> tuple(getGradFunction(t))).collect(Collectors.toList()));
	}

	protected Tensor toTensor(SymbolicTensor<TensorData, Size> tensor, String name, List<Tensor> tensors,
			List<TensorOperation<Tensor, Size>> operations) {
		boolean create_graph = tensors.stream().anyMatch(t -> t.create_graph());
		if (requires_grad) {
			Tensor t = createDefaultTensor(torch, tensor).withNextFunctions(name, operations,
					getGradFunctionTuples(tensors));

			return t.requires_grad_(True).create_graph_(create_graph);

		} else {
			return createDefaultTensor(torch, tensor).create_graph_(create_graph); 
		}
	}

	protected TensorData fromTensor(Tensor other) {

		if (other instanceof TensorBase) {
			return ((TensorBase) other).symbolicTensor.get();
		} else {
			if (other.size().getComponents().length > 2) {
				throw new IllegalArgumentException("More than 2 dimensions");
			}
			return other.toTensorData();
		}
	}

	@Override
	public TensorData toTensorData() {
		return fromTensor(this);
	}

	@Override
	public Tensor performUnaryMappingOperation(TensorOperation<TensorData, Size> operation,
			TensorOperation<Tensor, Size> backwardOp) {
		Size size = size();
		Tensor tensor = toTensor(performUnaryMappingOperation(new TensorOperationImpl<>(torch, operation.name(),
				t -> operation.apply(t), operation.sizeMapping(torch))), backwardOp.name(), backwardOp);
		tensor.size_(operation.sizeMapping(torch).apply(size));
		return tensor;
	}

	private SymbolicTensor<TensorData, Size> toSymbolicTensor(TensorData tensor) {
		return new SymbolicTensorAdapter<TensorData, Size>(tensor, tensor.size());
	}
	
	protected Tensor apply(Tensor other, DifferentiableTensorTensorFunction<TensorData> rowVectorOperation,
			DifferentiableTensorTensorFunction<TensorData> columnVectorOperation,
			DifferentiableTensorTensorFunction<TensorData> tensorOperation) {
		if (numel() != other.numel() && other.numel() != 1 && this.numel() != 1) {
			Size otherMatrixSize = other.size().asMatrixSize();
			if (otherMatrixSize.get(0) == 1) {
				return applyBinaryTensorOperation(rowVectorOperation, other);
			} else if (otherMatrixSize.get(1) == 1) {
				return applyBinaryTensorOperation(columnVectorOperation, other);
			} else {
				throw new IllegalStateException("Size doesn't match");
			}
		} else {
			return applyBinaryTensorOperation(tensorOperation, other);
		}
	}
	
	@Override
	public void backward(Tensor gradient, boolean create_graph) {
		if (gradient == null) {
			if (this.numel() == 1) {
				gradient = torch.ones(torch.Size(1, 1));
			} else {
				throw new RuntimeException("grad can be implicitly created only for scalar outputs");
			}
		} else {
			if (this.names() != null && this.names().length() > 0) {
				if (gradient.names() == null || gradient.names().length() == 0) {
					throw new IllegalArgumentException("Gradient names must not be null - should be " + this.names());
				} else {
					if (!SizeMatcher.isSizeMatch(this.size(), gradient.size())) {
						throw new IllegalArgumentException(
								"Names don't match:" + names() + " vs " + gradient.names());
					}
				}
			}
		}

		backwardRecursive(grad_fn(), gradient, create_graph);
	}

	@Override
	public void backward(Tensor gradient) {
		
		backward(gradient, create_graph);
	}

	private <T extends TensorOperations<T>> void backwardRecursive(GradFunction gf, Tensor back, boolean create_graph) {

		if (back == null) {
			throw new RuntimeException("backwardRecursive input tensor cannot be null");
		} else {
			if (back.size().dimensions().length > 0 && this.names() != null && this.names().length() > 0) {
				if (back.names() == null || back.names().length() == 0) {
					throw new IllegalArgumentException("backwardRecursive input tensor names must not be null - should be:" + this.names());
				}
			}
		}
		
		if ((pre_operation_requires_grad == null || pre_operation_requires_grad.booleanValue()) && !this.requires_grad() && this.grad_fn() == null) {
			throw new IllegalStateException("Tensor does not require grad and does not have a grad_fn");
		}

		if (gf != null) {
			if (gf.next_functions() != null && gf.next_functions().getComponents().length > 0) {
				int index = 0;

				for (TensorOperation<Tensor, Size> operation : gf.operations()) {
					Size originalSize = back.size();

					Tensor modifiedBack = operation.apply(back);
					if (operation.sizeMapping(torch).apply(originalSize).numel() != modifiedBack.size().numel()) {
						throw new IllegalStateException(
								"Size mapping size does not match the modified size of the tensor");
					}

					Tuple<GradFunction> gf2Tuple = gf.next_functions().get(index);
					GradFunction gf2 = gf2Tuple.get(0);
					backwardRecursive(gf2, modifiedBack, create_graph);
					index++;
				}
			} else {
				Tensor target = gf.variable();
				if (target != null) {
					
					
					if (target.numel() != back.numel() && !SizeMatcher.isSizeMatch(target.size(), back.size())) {
						throw new IllegalArgumentException(
								"Dimension names don't match:" + target.names() + " vs " + back.names());
					}
				
					if (target.grad() == null && target.requires_grad()) {
						if (create_graph) {
							var accumulated = torch.zeros(back.size())
									.requires_grad_(true).add(back);
							accumulated = accumulated.cloneTensor();;
							accumulated.create_graph_(create_graph);
							target.grad_(accumulated);
						} else {
							var accumulated = torch.zeros(back.size())
									.requires_grad_(false).add_(back);
							accumulated.create_graph_(false);
							target.grad_(accumulated);
						}
						
					}  else if (target.requires_grad()){
						if (create_graph) {
	
							var accumulated = target.grad().add(back);
							accumulated.create_graph_(true).requires_grad_(true);
							accumulated = accumulated.cloneTensor();;
							target.grad_(accumulated);
						} else {
							
							var accumulated = target.grad().requires_grad_(false).add_(back);
							accumulated.create_graph_(false).requires_grad_(false);
							target.grad_(accumulated);
						}
					}
				
					target.backward(back);
				}
			}
		}
	}

	@Override
	public Tensor grad_fn_(GradFunction grad_fn) {
		this.grad_fn = grad_fn;
		return this;
	}

	@Override
	public boolean requires_grad() {
		return requires_grad;
	}

	@Override
	public Tensor grad_(Tensor grad) {
		this.grad = grad;
		return this;
	}

	@Override
	public void backward() {
		backward(null);
	}

	@Override
	public Tensor grad() {
		return grad;
	}

	@Override
	public GradFunction grad_fn() {
		return grad_fn;
	}

	@Override
	public Tensor requires_grad_(boolean requires_grad) {
		this.requires_grad = requires_grad;
		return this;
	}

	@Override
	public Tensor names_(Tuple<String> names) {
		size().names_(names);

		return this;
	}

	@Override
	public Tuple<String> names() {
		return size().dimensionNames();
	}

	@Override
	public Tensor withNextFunctions(String name, List<TensorOperation<Tensor, Size>> operations,
			Tuple<Tuple<GradFunction>> nextFunctions) {
		if (self.grad_fn() != null) {
			throw new RuntimeException("Grad function already set");
		}
		this.grad_fn = new GradFunctionImpl(name, operations, null, nextFunctions);
		return this;
	}

	@Override
	public Tensor get() {
		return this;
	}

	@Override
	protected Tensor self() {
		return this;
	}

	@Override
	public void close() {

	}

	@Override
	public float item() {
		if (numel() != 1) {
			throw new IllegalStateException("only one element tensors can be converted to Java scalars");
		}
		return getDataAsFloatArray()[0];
	}

	@Override
	public boolean create_graph() {
		return create_graph;
	}

	@Override
	public Tensor create_graph_(boolean create_graph) {
		this.create_graph = create_graph;
		return this;
	}


}
