package org.jvmtorch.impl.ml4j;

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

	public static int LONG = 1;
	
	private MatrixFactory matrixFactory;
	private DirectedComponentsContext directedComponentsContext;
	
	public ML4JTorchImpl(DirectedComponentsContext directedComponentsContext, TensorDataConverter<ML4JTensorOperations> tensorDataConverter) {
		super(tensorDataConverter);
		this.directedComponentsContext = directedComponentsContext;
		this.matrixFactory = directedComponentsContext.getMatrixFactory();
	}
	
	private TensorData createTensorDataFromMatrix(Matrix matrix, Size size) {
		return tensorDataConverter.createTensorDataFromTensorOperations(new ML4JTensorOperationsImpl(this, directedComponentsContext, matrix, size));
	}
	
	private Tensor createTensorFromMatrix(Size size, String name, String inputName, Matrix matrix, boolean requires_grad) {
		return new ML4JTensor(this, directedComponentsContext, tensorDataConverter, name, inputName, createTensorDataFromMatrix(matrix, size), requires_grad);
	}
	
	public Tensor empty(Size size) {
		return createTensorFromMatrix(size, "empty","generatedempty", matrixFactory.createMatrix(size.asMatrixSize().getFirstComponent().numel(), size.asMatrixSize().getSecondComponent().numel()), false);
	}

	public Tensor zeros(Size size) {
		return createTensorFromMatrix(size, "zeros","generatedzeros", matrixFactory.createZeros(size.asMatrixSize().getFirstComponent().numel(), size.asMatrixSize().getSecondComponent().numel()), false);
	}
	
	public Tensor ones(Size size) {
		return createTensorFromMatrix(size, "ones","generatedones", matrixFactory.createOnes(size.asMatrixSize().getFirstComponent().numel(), size.asMatrixSize().getSecondComponent().numel()), false);
	}
	
	@Override
	public Tensor randn(Size size) {
		return createTensorFromMatrix(size, "randn","generatedrandn", matrixFactory.createRandn(size.asMatrixSize().getFirstComponent().numel(), size.asMatrixSize().getSecondComponent().numel()), false);
	}

	@Override
	public Tensor rand(Size size) {
		return createTensorFromMatrix(size, "rand","generatedrand", matrixFactory.createRand(size.asMatrixSize().getFirstComponent().numel(), size.asMatrixSize().getSecondComponent().numel()), false);
	}

	@Override
	public Tensor tensor(float[] data, Size size) {
		return createTensorFromMatrix(size, "tensor","data", matrixFactory.createMatrixFromRowsByRowsArray(size.asMatrixSize().getFirstComponent().numel(), size.asMatrixSize().getSecondComponent().numel(), data), false);
	}

	@Override
	public Tensor tensor(TensorData tensorOperations) {
		return new ML4JTensor(this, directedComponentsContext, tensorDataConverter,"name", "name", tensorOperations, false);
	}
}
