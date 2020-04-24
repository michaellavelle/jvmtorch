package org.jvmtorch.impl.ml4j;

import org.jvmtorch.impl.ScalarImpl;
import org.jvmtorch.impl.TorchImpl;
import org.jvmtorch.torch.Size;
import org.jvmtorch.torch.Tensor;
import org.jvmtorch.torch.TensorData;
import org.jvmtorch.torch.TensorDataConverter;
import org.jvmtorch.torch.Torch;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.components.DirectedComponentsContext;

public class ML4JTorchImpl extends TorchImpl<ML4JTensorOperations> implements Torch {
	
	private MatrixFactory matrixFactory;
	private DirectedComponentsContext directedComponentsContext;
	
	public ML4JTorchImpl(DirectedComponentsContext directedComponentsContext, TensorDataConverter<ML4JTensorOperations> tensorDataConverter) {
		super(tensorDataConverter);
		this.directedComponentsContext = directedComponentsContext;
		this.matrixFactory = directedComponentsContext.getMatrixFactory();
	}
	
	private TensorData createTensorDataFromMatrix(Matrix matrix, Size size) {
		TensorData data = tensorDataConverter.createTensorDataFromTensorOperations(new ML4JTensorOperationsImpl(this, directedComponentsContext, matrix, size));
		return data;
	}
	
	private Tensor createTensorFromMatrix(Size size, Matrix matrix, boolean requires_grad) {
		if (size.dimensions().length > 0) {
			return new ML4JTensor(this, directedComponentsContext, tensorDataConverter, createTensorDataFromMatrix(matrix, size), requires_grad);
		} else {
			return new ScalarImpl(this, matrix.get(0));
		}
	}
	
	public Tensor empty(Size size) {
		return createTensorFromMatrix(size, matrixFactory.createMatrix(size.asMatrixSize().getFirstComponent().numel(), size.asMatrixSize().getSecondComponent().numel()), false);
	}

	public Tensor zeros(Size size) {
		return createTensorFromMatrix(size, matrixFactory.createZeros(size.asMatrixSize().getFirstComponent().numel(), size.asMatrixSize().getSecondComponent().numel()), false);
	}
	
	public Tensor ones(Size size) {
		return createTensorFromMatrix(size, matrixFactory.createOnes(size.asMatrixSize().getFirstComponent().numel(), size.asMatrixSize().getSecondComponent().numel()), false);
	}
	
	@Override
	public Tensor randn(Size size) {
		return createTensorFromMatrix(size, matrixFactory.createRandn(size.asMatrixSize().getFirstComponent().numel(), size.asMatrixSize().getSecondComponent().numel()), false);
	}

	@Override
	public Tensor rand(Size size) {
		return createTensorFromMatrix(size,  matrixFactory.createRand(size.asMatrixSize().getFirstComponent().numel(), size.asMatrixSize().getSecondComponent().numel()), false);
	}

	@Override
	public Tensor tensor(float[] data, Size size) {
		return createTensorFromMatrix(size, matrixFactory.createMatrixFromRowsByRowsArray(size.asMatrixSize().getFirstComponent().numel(), size.asMatrixSize().getSecondComponent().numel(), data), false);
	}

	@Override
	public Tensor tensor(TensorData tensorOperations) {
		return new ML4JTensor(this, directedComponentsContext, tensorDataConverter, tensorOperations, false);
	}
}
