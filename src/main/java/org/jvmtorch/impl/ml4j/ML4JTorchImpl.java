package org.jvmtorch.impl.ml4j;

import org.jvmtorch.impl.ScalarImpl;
import org.jvmtorch.torch.Size;
import org.jvmtorch.torch.Tensor;
import org.jvmtorch.torch.TensorData;
import org.jvmtorch.torch.TensorDataConverter;
import org.jvmtorch.torch.Torch;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.components.DirectedComponentsContext;

public class ML4JTorchImpl implements Torch {

	public static int LONG = 1;
	
	private MatrixFactory matrixFactory;
	private DirectedComponentsContext directedComponentsContext;
	private TensorDataConverter<ML4JTensorOperations> tensorDataConverter;
	
	public ML4JTorchImpl(DirectedComponentsContext directedComponentsContext, TensorDataConverter<ML4JTensorOperations> tensorDataConverter) {
		this.directedComponentsContext = directedComponentsContext;
		this.matrixFactory = directedComponentsContext.getMatrixFactory();
		this.tensorDataConverter = tensorDataConverter;
	}
	
	private TensorData createTensorData(Matrix matrix, Size size) {
		return tensorDataConverter.createTensorDataFromTensorOperations(new ML4JTensorOperationsImpl(this, directedComponentsContext, matrix, size));
	}
	
	public Tensor empty(Size size) {
		return new ML4JTensor(this, directedComponentsContext, tensorDataConverter, "empty","generatedempty", createTensorData(matrixFactory.createMatrix(size.asMatrixSize().getFirstComponent().numel(), size.asMatrixSize().getSecondComponent().numel()), size), false);
	}

	public Tensor zeros(Size size) {
		return new ML4JTensor(this, directedComponentsContext, tensorDataConverter, "zeros", "generatedzeros",  createTensorData(matrixFactory.createZeros(size.asMatrixSize().getFirstComponent().numel(), size.asMatrixSize().getSecondComponent().numel()), size), false);
	}
	
	public Tensor ones(Size size) {
		return new ML4JTensor(this, directedComponentsContext, tensorDataConverter, "ones", "generatedones", createTensorData(matrixFactory.createOnes(size.asMatrixSize().getFirstComponent().numel(), size.asMatrixSize().getSecondComponent().numel()), size), false);
	}
	
	@Override
	public Tensor randn(Size size) {
		return new ML4JTensor(this, directedComponentsContext, tensorDataConverter, "randn","generatedrandn", createTensorData(matrixFactory.createRandn(size.asMatrixSize().getFirstComponent().numel(), size.asMatrixSize().getSecondComponent().numel()), size), false);
	}

	@Override
	public Tensor rand(Size size) {
		return new ML4JTensor(this, directedComponentsContext, tensorDataConverter, "rand", "generatedrand", createTensorData(matrixFactory.createRand(size.asMatrixSize().getFirstComponent().numel(), size.asMatrixSize().getSecondComponent().numel()), size), false);
	}

	@Override
	public Tensor tensor(float[] data, Size size) {
		return new ML4JTensor(this, directedComponentsContext, tensorDataConverter, "empty","generatedempty", createTensorData(matrixFactory.createMatrixFromRowsByRowsArray(size.asMatrixSize().getFirstComponent().numel(), size.asMatrixSize().getSecondComponent().numel(), data), size), false);
	}

	@Override
	public Tensor tensor(TensorData tensorOperations) {
		return new ML4JTensor(this, directedComponentsContext, tensorDataConverter,"name", "name", tensorOperations, false);
	}

	@Override
	public Tensor tensor(float value) {
		return new ScalarImpl(this, "scalar","scalar", value);
	}


}
