package org.jvmtorch.impl.ml4j;

import org.jvmtorch.torch.Size;
import org.jvmtorch.torch.TensorOperations;
import org.jvmpy.symbolictensors.Operatable;
import org.jvmpy.symbolictensors.Operation;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;

public class ML4JTensorOperations implements TensorOperations<ML4JTensorOperations>, Operatable<ML4JTensorOperations, ML4JTensorOperations> {

	private MatrixFactory matrixFactory;
	private Matrix matrix;
	
	public ML4JTensorOperations(MatrixFactory matrixFactory, Matrix matrix) {
		this.matrixFactory = matrixFactory;
		this.matrix = matrix;
		if (matrix.getRows() == 0 || matrix.getColumns() ==0) {
			throw new IllegalArgumentException();
		}
	}
	
	public Matrix getMatrix() {
		return matrix;
	}
	
	private ML4JTensorOperations toML4JTensorOperations(Matrix matrix) {
		return new ML4JTensorOperations(matrixFactory, matrix);
	}
	
	@Override
	public ML4JTensorOperations mul(float value) {
		return toML4JTensorOperations(matrix.asEditableMatrix().mul(value));
	}

	@Override
	public ML4JTensorOperations add(float value) {
		return toML4JTensorOperations(matrix.add(value));
	}

	@Override
	public ML4JTensorOperations sub_(ML4JTensorOperations mul) {
		matrix.asEditableMatrix().subi(mul.getMatrix());
		return this;
	}

	@Override
	public ML4JTensorOperations mul_(ML4JTensorOperations mul) {
		matrix.asEditableMatrix().muli(mul.getMatrix());
		return this;
	}

	@Override
	public ML4JTensorOperations add_(ML4JTensorOperations mul) {
		matrix.asEditableMatrix().addi(mul.getMatrix());
		return this;
	}

	@Override
	public ML4JTensorOperations matmul(ML4JTensorOperations other) {



		return toML4JTensorOperations(matrix.mmul(other.getMatrix()));
	}

	@Override
	public ML4JTensorOperations mul(ML4JTensorOperations other) {
		return toML4JTensorOperations(matrix.mul(other.getMatrix()));
	}

	@Override
	public int numel() {
		return matrix.getLength();
	}

	@Override
	public ML4JTensorOperations add(ML4JTensorOperations other) {
		return toML4JTensorOperations(matrix.add(other.getMatrix()));
	}

	@Override
	public ML4JTensorOperations mean() {
		return toML4JTensorOperations(matrixFactory.createOnes(1, 1).mul(matrix.sum() / matrix.getLength()));
	}

	@Override
	public ML4JTensorOperations transpose() {
		return toML4JTensorOperations(matrix.transpose());
	}

	@Override
	public ML4JTensorOperations t() {
		return toML4JTensorOperations(matrix.transpose());
	}

	@Override
	public Size size() {

		if (matrix.getColumns() == 0) {
			throw new IllegalStateException();
		}

		return new Size(matrix.getRows(), matrix.getColumns());
	}

	@Override
	public ML4JTensorOperations get() {
		return this;
	}

	@Override
	public void performInlineOperation(Operation<ML4JTensorOperations> operation) {
		operation.apply(this);
	}

	@Override
	public ML4JTensorOperations performUnaryMappingOperation(String newTensorName, Operation<ML4JTensorOperations> operation) {
		return 	operation.apply(this);
	}

	@Override
	public float[] data() {
		return matrix.getRowByRowArray();
	}
}
