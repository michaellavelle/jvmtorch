package org.jvmtorch.impl;

import org.jvmtorch.impl.ml4j.ML4JTensor;
import org.jvmtorch.impl.ml4j.ML4JTensorOperations;
import org.jvmtorch.torch.Tensor;
import org.jvmtorch.torch.Torch;
import org.ml4j.MatrixFactory;

public class TorchImpl implements Torch<ML4JTensorOperations> {

	public static int LONG = 1;
	
	private MatrixFactory matrixFactory;
	
	public TorchImpl(MatrixFactory matrixFactory) {
		this.matrixFactory = matrixFactory;
	}
	
	public Tensor<ML4JTensorOperations> empty(int i, int j) {
		if (i == 0 || j == 0) {
			throw new IllegalArgumentException();
		}
		return new ML4JTensor(this, matrixFactory, "empty","generatedempty",matrixFactory.createMatrix(i, j));
	}

	public Tensor<ML4JTensorOperations> zeros(int i, int j) {
		if (i == 0 || j == 0) {
			throw new IllegalArgumentException();
		}
		return new ML4JTensor(this, matrixFactory, "zeros", "generatedzeros",matrixFactory.createZeros(i, j));
	}
	
	public Tensor<ML4JTensorOperations> ones(int i, int j) {
		if (i == 0 || j == 0) {
			throw new IllegalArgumentException();
		}
		return new ML4JTensor(this, matrixFactory, "ones", "generatedones",matrixFactory.createOnes(i, j));
	}

	@Override
	public Tensor<ML4JTensorOperations> randn(int i, int j, int k, int l) {
		// TODO
		if (i == 0 || j == 0 || k == 0 || l == 0) {
			throw new IllegalArgumentException();
		}
		return new ML4JTensor(this, matrixFactory, "randn","generatedrandn",matrixFactory.createRandn(i, j));
	}

	@Override
	public Tensor<ML4JTensorOperations> rand(int i, int j) {
		if (i == 0 || j == 0) {
			throw new IllegalArgumentException();
		}
		return new ML4JTensor(this, matrixFactory, "rand", "generatedrand",matrixFactory.createRand(i, j));
	}

	@Override
	public Tensor<ML4JTensorOperations> randn(int i, int j) {
		if (i == 0 || j == 0) {
			throw new IllegalArgumentException();
		}
		return new ML4JTensor(this, matrixFactory, "randn", "generatedrandn",matrixFactory.createRandn(i, j));
	}

	@Override
	public Tensor<ML4JTensorOperations> randn(int i) {
		if (i == 0) {
			throw new IllegalArgumentException();
		}
		return new ML4JTensor(this, matrixFactory, "randn", "generatedrandn",matrixFactory.createRandn(1, i));
	}

	@Override
	public Tensor<ML4JTensorOperations> randn(int... sizes) {
		for (int i : sizes) {
			if (i == 0) {
				throw new IllegalArgumentException();
			}
		}

		return new ML4JTensor(this, matrixFactory, "randn", "generatedrandn",matrixFactory.createRandn(sizes[0], sizes[1]));
	}
}
