package org.jvmtorch.impl.ml4j;

import java.util.function.BiFunction;

import org.jvmtorch.JvmTorchFactory;
import org.jvmtorch.impl.OptimImpl;
import org.jvmtorch.nn.NN;
import org.jvmtorch.nn.functional.Functional;
import org.jvmtorch.torch.Size;
import org.jvmtorch.torch.TensorConverter;
import org.jvmtorch.torch.TensorDataConverter;
import org.jvmtorch.torch.Torch;
import org.jvmtorch.torch.optim.Optim;
import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.DirectedComponentsContextImpl;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.quickstart.sessions.factories.QuickstartSessionFactory;

public class ML4JJvmTorchFactory implements JvmTorchFactory {
	
    public static final MatrixFactory DEFAULT_MATRIX_FACTORY = new JBlasRowMajorMatrixFactory();
    
    public static final DirectedComponentsContext DEFAULT_DIRECTED_COMPONENTS_CONTEXT = new DirectedComponentsContextImpl(DEFAULT_MATRIX_FACTORY, false);

    public static final DirectedComponentFactory DEFAULT_DIRECTED_COMPONENT_FACTORY = 
    		new QuickstartSessionFactory(DEFAULT_MATRIX_FACTORY, false).getNeuralComponentFactory();
	
	private DirectedComponentsContext directedComponentsContext;
	private DirectedComponentFactory directedComponentFactory;
	private TensorDataConverter<ML4JTensorOperations> tensorDataConverter;
	private TensorConverter<ML4JTensor> ml4jTensorConverter;
	
	public ML4JJvmTorchFactory() {
		this.directedComponentsContext = DEFAULT_DIRECTED_COMPONENTS_CONTEXT;
		this.directedComponentFactory = DEFAULT_DIRECTED_COMPONENT_FACTORY;
		this.tensorDataConverter = new TensorDataConverter<>(ML4JTensorOperations.class, new BiFunction<float[], Size, ML4JTensorOperations>() {

			@Override
			public ML4JTensorOperations apply(float[] t, Size size) {
				Size matrixSize = size.asMatrixSize();
				return new ML4JTensorOperationsImpl(createTorch(), directedComponentsContext, 
						directedComponentsContext.getMatrixFactory().createMatrixFromRowsByRowsArray(matrixSize.get(0), matrixSize.get(1), t), size);
			}}
		);

		this.ml4jTensorConverter = new ML4JTensorConverter(tensorDataConverter, directedComponentsContext);
	}

	@Override
	public Torch createTorch() {
		return new ML4JTorchImpl(directedComponentsContext, tensorDataConverter);
	}

	@Override
	public Functional createFunctional() {
		return new ML4JFunctionalImpl(directedComponentFactory, tensorDataConverter,  ml4jTensorConverter);
	}

	@Override
	public NN createNN() {
		return new ML4JNNImpl(directedComponentsContext, directedComponentFactory, 
				createTorch(), createFunctional(), tensorDataConverter, ml4jTensorConverter);
	}

	@Override
	public Optim createOptim() {
		return new OptimImpl(createTorch());
	}

}
