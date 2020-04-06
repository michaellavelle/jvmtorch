package org.jvmtorch.impl.ml4j;

import org.jvmtorch.JvmTorchFactory;
import org.jvmtorch.impl.OptimImpl;
import org.jvmtorch.impl.TorchImpl;
import org.jvmtorch.nn.Functional;
import org.jvmtorch.nn.NN;
import org.jvmtorch.torch.Torch;
import org.jvmtorch.torch.optim.Optim;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.components.factories.DirectedComponentFactory;

public class ML4JJvmTorchFactory implements JvmTorchFactory<ML4JTensorOperations> {

	private MatrixFactory matrixFactory;
	private DirectedComponentFactory directedComponentFactory;
	
	public ML4JJvmTorchFactory(DirectedComponentFactory directedComponentFactory, MatrixFactory matrixFactory) {
		this.matrixFactory = matrixFactory;
		this.directedComponentFactory = directedComponentFactory;
	}

	@Override
	public Torch<ML4JTensorOperations> createTorch() {
		return new TorchImpl(matrixFactory);
	}

	@Override
	public Functional<ML4JTensorOperations> createFunctional() {
		return new ML4JFunctionalImpl(directedComponentFactory, matrixFactory);
	}

	@Override
	public NN<ML4JTensorOperations> createNN() {
		return new ML4JNNImpl(matrixFactory, directedComponentFactory, createTorch(), createFunctional());
	}

	@Override
	public Optim<ML4JTensorOperations> createOptim() {
		return new OptimImpl<>();
	}

}
