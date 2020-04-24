package org.jvmtorch.impl;

import java.util.function.UnaryOperator;

import org.jvmpy.symbolictensors.SymbolicTensor;
import org.jvmtorch.impl.operations.tensorscalar.ScalarAddition;
import org.jvmtorch.impl.operations.tensorscalar.ScalarMultiplication;
import org.jvmtorch.impl.operations.tensorscalar.ScalarSubtraction;
import org.jvmtorch.impl.operations.tensortensor.ColumnVectorAddition;
import org.jvmtorch.impl.operations.tensortensor.RowVectorAddition;
import org.jvmtorch.impl.operations.tensortensor.TensorAddition;
import org.jvmtorch.impl.operations.tensortensor.TensorMatrixMultiplication;
import org.jvmtorch.impl.operations.tensortensor.TensorMultiplication;
import org.jvmtorch.impl.operations.tensortensor.TensorSubtraction;
import org.jvmtorch.torch.Size;
import org.jvmtorch.torch.Tensor;
import org.jvmtorch.torch.TensorData;
import org.jvmtorch.torch.TensorDataConverter;
import org.jvmtorch.torch.Torch;

public class TensorImpl extends TensorBase {

	public TensorImpl(Torch torch, TensorDataConverter<?> tensorDataConverter,
			SymbolicTensor<TensorData, Size> symbolicTensor) {
		super(torch, tensorDataConverter, symbolicTensor);
	}

	public TensorImpl(Torch torch, TensorDataConverter<?> tensorDataConverter, TensorData tensorData) {
		super(torch, tensorDataConverter, tensorData);
	}
	
	@Override
	protected Tensor createDefaultTensor(Torch torch, SymbolicTensor<TensorData, Size> tensor) {
		return new TensorImpl(torch, tensorDataConverter, tensor);
	}
	
	// Unary Tensor operations

	@Override
	public Tensor t() {
		return applyUnaryTensorOperation("T", t -> t.t(), s -> s.t(),  g -> g.t());
	}
	
	@Override
	public Tensor sum() {
		return applyUnaryTensorOperation("Sum", t -> t.sum(), s -> torch.Size(1, 1), g -> g.mul(torch.ones(size())));
	}
	
	@Override
	public Tensor mean() {
		return applyUnaryTensorOperation("Mean", t -> t.mean(), s -> torch.Size(1, 1), g -> g.mul(torch.ones(size())).mul(1f / numel()));
	}
	
	@Override
	public Tensor norm() {
		return applyUnaryTensorOperation("Norm", t -> t.norm(), s -> size(), normBackward());
	}
	
	private UnaryOperator<Tensor> normBackward() {
		return t -> { throw new UnsupportedOperationException("Not yet implemented"); };
	}
	
	@Override
	public Tensor cloneTensor() {
		return applyUnaryTensorOperation("Clone", t -> t.cloneTensor(), s -> size(), t -> t);
	}

	@Override
	public Tensor columnSums() {
		return applyUnaryTensorOperation("ColumnSums", t -> t.columnSums(), 
				s -> torch.Size(1, size().get(1)), t -> torch.ones(size()).mul(t));
	
	}

	@Override
	public Tensor rowSums() {
		return applyUnaryTensorOperation("RowSums", t -> t.rowSums(), 
				s -> torch.Size(size().get(0), 1), t -> torch.ones(size()).mul(t));
	}


	@Override
	public Tensor view(Size size) {
		return applyUnaryTensorOperation("View", t -> t.view(size), 
				s -> size, t -> view(size()));

	}
	
	// Binary Tensor operations
	

	@Override
	public Tensor mul(Tensor other) {
		// TODO - Segment into different operations for broadcast shapes
		return applyBinaryTensorOperation(new TensorMultiplication<>(torch), other);
	}
	
	@Override
	public Tensor matmul(Tensor other) {
		return applyBinaryTensorOperation(new TensorMatrixMultiplication<>(torch), other);
	}
	
	@Override
	public Tensor add(Tensor other) {
		return apply(other, new RowVectorAddition<>(torch), new ColumnVectorAddition<>(torch),
				new TensorAddition<>(torch));
	}
	
	@Override
	public Tensor div(Tensor other) {
		// TODO - Segment into different operations for broadcast shapes
		return applyBinaryTensorOperation(new TensorAddition<>(torch), other);
	}

	@Override
	public Tensor sub(Tensor other) {
		// TODO - Segment into different operations for broadcast shapes
		return applyBinaryTensorOperation(new TensorSubtraction<>(torch), other);
	}
	
	
	// Binary Tensor-Scalar operations.
	
	@Override
	public Tensor mul(float value) {
		return applyScalarOperation(new ScalarMultiplication<>(), value);
	}

	@Override
	public Tensor add(float value) {
		return applyScalarOperation(new ScalarAddition<>(), value);
	}

	@Override
	public Tensor sub(float value) {
		return applyScalarOperation(new ScalarSubtraction<>(), value);
	}
	
	// Inline Tensor-Tensor operations
	
	@Override
	public Tensor sub_(Tensor other) {
		return applyInlineOperation(t -> t.sub_(fromTensor(other)));
	}

	@Override
	public Tensor mul_(Tensor other) {
		return applyInlineOperation(t -> t.mul_(fromTensor(other)));
	}

	@Override
	public Tensor add_(Tensor other) {
		return applyInlineOperation(t -> t.add_(fromTensor(other)));
	}


}
