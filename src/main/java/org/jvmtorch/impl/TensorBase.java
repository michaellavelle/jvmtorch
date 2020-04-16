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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.jvmpy.python.PythonClass;
import org.jvmpy.python.Tuple;
import org.jvmpy.symbolictensors.InitialSymbolicTensorImpl;
import org.jvmpy.symbolictensors.SymbolicTensor;
import org.jvmtorch.impl.operations.tensorscalar.DifferentiableTensorScalarFunction;
import org.jvmtorch.impl.operations.tensorscalar.ScalarAddition;
import org.jvmtorch.impl.operations.tensorscalar.ScalarMultiplication;
import org.jvmtorch.impl.operations.tensorscalar.TensorScalarImpl;
import org.jvmtorch.impl.operations.tensortensor.DifferentiableTensorTensorFunction;
import org.jvmtorch.impl.operations.tensortensor.TensorAddition;
import org.jvmtorch.impl.operations.tensortensor.TensorMatrixMultiplication;
import org.jvmtorch.impl.operations.tensortensor.TensorMultiplication;
import org.jvmtorch.torch.GradFunction;
import org.jvmtorch.torch.Size;
import org.jvmtorch.torch.Tensor;
import org.jvmtorch.torch.TensorData;
import org.jvmtorch.torch.TensorDataConverter;
import org.jvmtorch.torch.TensorOperation;
import org.jvmtorch.torch.TensorOperations;
import org.jvmtorch.torch.Torch;


public class TensorBase extends PythonClass<Tensor> implements Tensor {

	protected SymbolicTensor<TensorData> symbolicTensor;
	protected Torch torch;
	protected Tensor grad;
	protected GradFunction grad_fn;
	protected boolean requires_grad;
	protected Size size;
	protected TensorDataConverter<?> tensorDataConverter;
	
	public Torch torch() {
		return torch;
	}

	public TensorBase(Torch torch, TensorDataConverter<?> tensorDataConverter, SymbolicTensor<TensorData> symbolicTensor) {
		this.torch = torch;
		this.symbolicTensor = symbolicTensor;
		this.size = torch.Size(symbolicTensor.dimensions());
		if (symbolicTensor.dimensionNames() != null) {
			this.size.names_(tuple(symbolicTensor.dimensionNames()));
		}
		this.tensorDataConverter = tensorDataConverter;
	}

	public TensorBase(Torch torch, TensorDataConverter<?> tensorDataConverter, String name, String inputName, TensorData tensorData) {
		this.torch = torch;
		this.symbolicTensor = toSymbolicTensor(name, inputName, tensorData);
		this.size = tensorData.size();
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
		} else if (grad_fn()  != null) {
			if (names() != null && names().length() > 0) {
				return "tensor(name='" + this.symbolicTensor.getName() + "', "  + s.toString() + ", grad_fn=" + grad_fn().toString() + ", names=" + names() + ")";
			} else {
				return "tensor(name='" + this.symbolicTensor.getName() + "', "  + s.toString() + ", grad_fn=" + grad_fn().toString() + ")";
			}
		}
		else {
			if (names() != null && names().length() > 0) {
				return "tensor(name='" + this.symbolicTensor.getName()+ "', "  + s.toString() + ", names=" + names() + ")";
			} else {
				return "tensor(name='" + this.symbolicTensor.getName()+ "', "  + s.toString() + ")";

			}
		}
	}

	@Override
	public float[] getDataAsFloatArray() {
		float[] data =  this.symbolicTensor.evaluate().getDataAsFloatArray();
		if (data.length != this.numel()) {
			throw new IllegalStateException();
		}
		return data;
	}
	
	public Tensor size_(Size size) {
		Size existingSize = size();
		if (existingSize.numel() != size.numel()) {
			throw new IllegalArgumentException(existingSize.numel() + ":" + size.numel());
		}
		this.size = size;
		return this;
	}
	
	protected Tensor createDefaultTensor(Torch torch, SymbolicTensor<TensorData> tensor) {
		return new TensorBase(torch, tensorDataConverter, tensor);
	}
	
	@Override
	public Size size() {
		if (size == null) {
			size = torch.Size(symbolicTensor.dimensions()).names_(getDimensionNames(symbolicTensor.dimensionNames()));
		}
		return size;
	}

	@Override
	public Tensor mul(float value) {
		return applyScalarOperation(new ScalarMultiplication<>(), value);
	}
	
	@Override
	public Tensor view(int i, int j) {

		if (i == -1 && j == -1) {
			throw new RuntimeException("only one dimension can be inferred");
		} else {
			if (i == -1) {
				i = this.numel() / j;
			}
			if (j == -1) {
				j = this.numel() / i;
			}
		}
		
		if (names() != null && names().length() != 0) {
			throw new RuntimeException("Names not yet supported with named tensors");
		}
		
		final int finalI = i;
		final int finalJ = j;

		return toTensor(symbolicTensor
				.performUnaryMappingOperation("view", 
						new TensorOperationImpl<TensorData>(torch, "view", t -> t.view(finalI, finalJ),
								s-> torch.Size(finalI, finalJ))), "view"+ "Backward", 
				new TensorOperationImpl<Tensor>(torch, "viewBackward", 
						l -> viewBackward(l), s -> size()));

	}
	
	@Override
	public Tensor view(Size size) {
		return toTensor(symbolicTensor
				.performUnaryMappingOperation("view", 
						new TensorOperationImpl<TensorData>(torch, "view", t -> t.view(size),
								s-> size)), "view"+ "Backward", 
				new TensorOperationImpl<Tensor>(torch, "viewBackward", 
						l -> viewBackward(l), s -> size()));

	}

	private Tensor viewBackward(Tensor l) {
		return l.view(size());
	}
	
	private List<String> getNames(Size size) {
		return size.dimensionNames() == null ? null : size.dimensionNames().asList();
	}

	@Override
	public Tensor add(float value) {
		return applyScalarOperation(new ScalarAddition<>(), value);
	}
	
	@Override
	public Tensor mul(Tensor other) {
		return applyTensorOperation(new TensorMultiplication<>(torch), other);
	}
	
	@Override
	public Tensor add(Tensor other) {
		return applyTensorOperation(new TensorAddition<>(torch), other);
	}
	
	@Override
	public Tensor sub_(Tensor other) {
		symbolicTensor
		.performInlineOperation(new TensorOperationImpl<>(torch, "Sub", t -> { t.sub_(fromTensor(other)); return t; }, s -> s));
		return this;
	}

	@Override
	public Tensor mul_(Tensor other) {
		symbolicTensor
				.performInlineOperation(new TensorOperationImpl<>(torch, "Mul", t -> { t.mul_(fromTensor(other)); return t; }, s -> s));
		return this;
	}

	@Override
	public Tensor add_(Tensor other) {
		symbolicTensor
		.performInlineOperation(new TensorOperationImpl<>(torch, "Add", t -> { t.add_(fromTensor(other)); return t; }, s -> s));
		return this;
	}
	

	@Override
	public Tensor t() {
			
		return toTensor(symbolicTensor
				.performUnaryMappingOperation("T", new TensorOperationImpl<>(torch, "Transpose",t -> t.t(), s -> size().t())), "TBackward", new TensorOperationImpl<>(torch, "TBackward", t -> t.t(), s -> s.t()));
	}


	@Override
	public Tensor mean() {
		// TODO backward
		return toTensor(symbolicTensor
				.performUnaryMappingOperation("Mean", new TensorOperationImpl<>(torch, "Mean", t -> t.mean(), s -> torch.Size(1, 1))), "MeanBackward", new TensorOperationImpl<>(torch, "MeanBackward", g -> g.mul(1f / numel()), s -> size()));
	}
	

	@Override
	public Tensor matmul(Tensor other) {
		return applyTensorOperation(new TensorMatrixMultiplication<>(torch), other);

	}
	
	private Tensor applyTensorOperation(DifferentiableTensorTensorFunction<TensorData> tensorOperation, Tensor other) {
				
		UnaryOperator<TensorData> forwardPropFunction = tensorOperation.forwardPropFunction(fromTensor(other));
		
		Pair<UnaryOperator<Tensor>, UnaryOperator<Tensor>> backPropFunctions = tensorOperation.backPropFunctions(new ImmutablePair<>(this, other));
		
		UnaryOperator<Size> sizeFunction = tensorOperation.sizeFunction(new ImmutablePair<>(size(), other.size()));
		
		if (other.requires_grad()) {
			return toTensor(symbolicTensor
					.performUnaryMappingOperation(tensorOperation.name(), new TensorOperationImpl<>(torch, tensorOperation.name(), forwardPropFunction, sizeFunction)), tensorOperation.name() + "Backward", Arrays.asList(this, other), Arrays.asList(new TensorOperationImpl<>(torch, tensorOperation.name() + "Backward", backPropFunctions.getLeft(), s -> size()), new TensorOperationImpl<>(torch, tensorOperation.name() + "Backward", backPropFunctions.getRight(), s -> other.size())));
		} else {
			return toTensor(symbolicTensor
					.performUnaryMappingOperation(tensorOperation.name(), new TensorOperationImpl<>(torch, tensorOperation.name(), forwardPropFunction, sizeFunction)), tensorOperation.name() + "Backward", new TensorOperationImpl<>(torch, tensorOperation.name() + "Backward", backPropFunctions.getLeft(), s -> s));
		}
	}
	
	private Tensor applyScalarOperation(DifferentiableTensorScalarFunction<TensorData> tensorOperation, float scalar) {
		
		UnaryOperator<TensorData> forwardPropFunction = tensorOperation.forwardPropFunction(scalar);
		
		UnaryOperator<Tensor> backPropFunction = tensorOperation.backPropFunction(new TensorScalarImpl<>(this, scalar));
		
		return toTensor(symbolicTensor
					.performUnaryMappingOperation(tensorOperation.name(), new TensorOperationImpl<>(torch, tensorOperation.name(), forwardPropFunction, s -> s)), tensorOperation.name() + "Backward", new TensorOperationImpl<>(torch, tensorOperation.name() + "Backward", backPropFunction, s -> s));

	}


	@Override
	public int numel() {
		
		return Math.max(Arrays.stream(symbolicTensor.dimensions()).reduce(1, (a, b) -> a * b), 1);
	}

	

	protected Tensor toTensor(SymbolicTensor<TensorData> tensor, String name, TensorOperation<Tensor> operation) {
		if (this.requires_grad()) {
			if (this.grad_fn() != null) {
				Tensor t = createDefaultTensor(torch, tensor)
						.withNextFunctions(name, Arrays.asList(operation),
								tuple(tuple(this.grad_fn()))
						);
				return t.requires_grad_(True).names_(getDimensionNames(tensor.dimensionNames()));
			} else {
		
				if  (tensor.dimensionNames() != null) {
				Tensor t = createDefaultTensor(torch, tensor)
						.withNextFunctions(name, Arrays.asList(operation),
								tuple(tuple(new GradFunctionImpl("AccumulateGrad", Arrays.asList(new TensorOperationImpl<>(torch, "AccumulateGrad", l -> l, s-> s)),
										this, null)))
						);

				return t.requires_grad_(True).names_(getDimensionNames(tensor.dimensionNames()));
				
				} else {
					Tensor t = createDefaultTensor(torch, tensor)
							.withNextFunctions(name, Arrays.asList(operation),
									tuple(tuple(new GradFunctionImpl("AccumulateGrad", Arrays.asList(new TensorOperationImpl<>(torch, "AccumulateGrad", l -> l, s -> s)),
											this, null)))
							);

					return t.requires_grad_(True);
					
				}
			}

		} else {
			return createDefaultTensor(torch, tensor).names_(getDimensionNames(tensor.dimensionNames()));
		}
	}


	private GradFunction getGradFunction(Tensor tensor) {
		if (tensor.grad_fn() != null) {
			return tensor.grad_fn();
		} else {
			return new GradFunctionImpl("AccumulateGrad", Arrays.asList(new TensorOperationImpl<>(torch, "AccumulateGrad", l -> l, s -> s)),
					tensor, null);
		}
	}
	
	private Tuple<Tuple<GradFunction>> getGradFunctionTuples(List<Tensor> tensors) {
		return tuple(tensors.stream().map(t -> tuple(getGradFunction(t))).collect(Collectors.toList()));
	}
	
	private Tuple<String> getDimensionNames(List<String> names) {
		return names == null ? null : tuple(names);
	}

	protected Tensor toTensor(SymbolicTensor<TensorData> tensor, String name, List<Tensor> tensors, List<TensorOperation<Tensor>> operations) {
		if (this.requires_grad()) {
				Tensor t = createDefaultTensor(torch, tensor)
						.withNextFunctions(name, operations,
								getGradFunctionTuples(tensors));

				return t.requires_grad_(True).names_(getDimensionNames(tensor.dimensionNames()));
						
		} else {
			return createDefaultTensor(torch, tensor).names_(getDimensionNames(tensor.dimensionNames()));
		}
	}

	protected TensorData fromTensor(Tensor other) {
		
		if(other instanceof TensorBase){
            return ((TensorBase)other).symbolicTensor.get();
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
	public Tensor performUnaryMappingOperation(String newTensorName, TensorOperation<TensorData> operation, TensorOperation<Tensor> backwardOp) {
		Size size = size();
		Tensor tensor = toTensor(symbolicTensor
				.performUnaryMappingOperation(newTensorName, new TensorOperationImpl<>(torch, operation.name(), t -> operation.apply(t), operation.sizeMapping(torch))),backwardOp.name(), backwardOp);
		tensor.size_(operation.sizeMapping(torch).apply(size));
		return tensor;
	}

	private SymbolicTensor<TensorData> toSymbolicTensor(String name, String inputName, TensorData tensor) {
		return new InitialSymbolicTensorImpl<TensorData>(name, inputName, () -> tensor, tensor.size().dimensions(), getNames(tensor.size()));
	}


	@Override
	public void backward(Tensor gradient) {
		if (gradient == null) {
			if (this.numel() == 1) {
				gradient = torch.ones(torch.Size(1, 1));
			} else {
				throw new RuntimeException("grad can be implicitly created only for scalar outputs");
			}
		} else {
			if (this.names() != null && this.names().length() > 0) {
				if (gradient.names() == null || gradient.names().length() == 0) {
					throw new IllegalArgumentException("Gradient names must not be null");
				} else {
					if (!toScopeIndependentNamesList(this.names().asList()).equals(toScopeIndependentNamesList(gradient.names().asList()))) {	
						throw new IllegalArgumentException("Names don't match:" + this.names() + " vs " + gradient.names());
					}
				}
			}
		}

		backwardRecursive(grad_fn(), gradient);

	}


	private List<String> toScopeIndependentNamesList(List<String> strings) {
		List<String> returnValues = new ArrayList<>();
		for (String s : strings) {
			s = s.replaceAll("input_", "");
			s = s.replaceAll("output_", "");
			returnValues.add(s);
		}
		
		return returnValues;
	}

	public <T extends TensorOperations<T>> void backwardRecursive(GradFunction gf, Tensor back) {

		if (back == null) {
			throw new RuntimeException("backwardRecursive input tensor cannot be null");
		} else {
			if (this.names() != null && this.names().length() > 0) {
				if (back.names() == null || back.names().length() == 0) {
					throw new IllegalArgumentException("backwardRecursive input tensor names must not be null");
				} 
			}
		}

		if (gf != null) {
			if (gf.next_functions() != null && gf.next_functions().getComponents().length > 0) {
				int index = 0;

				for (UnaryOperator<Tensor> operation : gf.operations()) {
					Size originalSize = back.size();
					Tensor back2 = operation.apply(back);
					if (operation instanceof TensorOperation) {
						TensorOperation<?> op = (TensorOperation<?>)operation;
						back2.size_(op.sizeMapping(torch).apply(originalSize));
					}
					Tuple<GradFunction> gf2Tuple = gf.next_functions().get(index);
					GradFunction gf2 = gf2Tuple.get(0);
					backwardRecursive(gf2, back2);
					index++;
				}
			} else {
				Tensor target = gf.variable();
				if (target != null) {
					if (!toScopeIndependentNamesList(target.names().asList()).equals(toScopeIndependentNamesList(back.names().asList()))) {
						throw new IllegalArgumentException("Names don't match:" + target.names() + " vs " + back.names());
					}
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
		this.requires_grad = true;
		return this;
	}
	
	@Override
	public Tensor names_(Tuple<String> names) {
		size().names_(names);
		
		return this;
	}
	
	@Override
	public Tuple<String> names() {
		return size.dimensionNames();
	}

	@Override
	public Tensor withNextFunctions(String name, List<TensorOperation<Tensor>> operations, Tuple<Tuple<GradFunction>> nextFunctions) {
		if (self.grad_fn()  != null) {
			throw new RuntimeException("Already set");
		}
		this.grad_fn = new GradFunctionImpl(name,  operations, null, nextFunctions);
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
}
