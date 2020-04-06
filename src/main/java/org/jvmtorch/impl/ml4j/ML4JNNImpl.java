package org.jvmtorch.impl.ml4j;

import org.jvmtorch.impl.NNImpl;
import org.jvmtorch.nn.Conv2d;
import org.jvmtorch.nn.Functional;
import org.jvmtorch.nn.Linear;
import org.jvmtorch.nn.modules.MSELoss;
import org.jvmtorch.torch.Torch;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.components.factories.DirectedComponentFactory;

public class ML4JNNImpl extends NNImpl<ML4JTensorOperations> {

	private DirectedComponentFactory directedComponentFactory;
	private MatrixFactory matrixFactory;

	
    public ML4JNNImpl(MatrixFactory matrixFactory, DirectedComponentFactory directedComponentFactory, Torch<ML4JTensorOperations> torch, Functional<ML4JTensorOperations> functional) {
        super(torch, functional);
        this.directedComponentFactory = directedComponentFactory;
        this.matrixFactory = matrixFactory;
    }

    @Override
    public Conv2d<?, ML4JTensorOperations> Conv2d(int... params) {
        return new ML4JConv2d(this, params[0], params[1], params[2]);
    }

    public DirectedComponentFactory getDirectedComponentFactory() {
        return directedComponentFactory;
    }

    public MatrixFactory getMatrixFactory() {
        return matrixFactory;
    }

    @Override
    public Linear<?, ML4JTensorOperations> Linear(int... params) {
        return new ML4JLinear(this, params[0], params[1]);
    }

    @Override
    public MSELoss<ML4JTensorOperations> MSELoss() {
        return new ML4JMSELossImpl(torch);
    }
}
