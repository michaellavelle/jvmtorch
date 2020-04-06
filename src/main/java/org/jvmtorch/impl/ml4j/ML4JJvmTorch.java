package org.jvmtorch.impl.ml4j;

import org.jvmtorch.JvmTorchFactory;
import org.jvmtorch.nn.Functional;
import org.jvmtorch.nn.NN;
import org.jvmtorch.torch.Torch;
import org.jvmtorch.torch.optim.Optim;
import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.quickstart.sessions.factories.QuickstartSessionFactory;

public class ML4JJvmTorch {

    public static final MatrixFactory DEFAULT_MATRIX_FACTORY;

    public static final DirectedComponentFactory DEFAULT_DIRECTED_COMPONENT_FACTORY;


    public static final JvmTorchFactory<ML4JTensorOperations> ML4J_PYTORCH_FACTORY;

    public static final Torch<ML4JTensorOperations> torch;
    public static final Functional<ML4JTensorOperations> F;
    public static final NN<ML4JTensorOperations> nn;
    public static final Optim<ML4JTensorOperations> optim;

    static {
        DEFAULT_MATRIX_FACTORY = new JBlasRowMajorMatrixFactory();
        DEFAULT_DIRECTED_COMPONENT_FACTORY = new QuickstartSessionFactory(DEFAULT_MATRIX_FACTORY, false).getNeuralComponentFactory();
        ML4J_PYTORCH_FACTORY = new ML4JJvmTorchFactory(DEFAULT_DIRECTED_COMPONENT_FACTORY, DEFAULT_MATRIX_FACTORY);
        torch = ML4J_PYTORCH_FACTORY.createTorch();
        F = ML4J_PYTORCH_FACTORY.createFunctional();
        nn = ML4J_PYTORCH_FACTORY.createNN();
        optim = ML4J_PYTORCH_FACTORY.createOptim();
    }
}
