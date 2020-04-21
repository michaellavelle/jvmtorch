package org.jvmtorch.impl.ml4j;

import org.jvmpy.symbolictensors.SymbolicTensor;
import org.jvmtorch.impl.ScalarImpl;
import org.jvmtorch.impl.TensorBase;
import org.jvmtorch.torch.Size;
import org.jvmtorch.torch.Tensor;
import org.jvmtorch.torch.TensorData;
import org.jvmtorch.torch.TensorDataConverter;
import org.jvmtorch.torch.Torch;
import org.ml4j.Matrix;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.ml4j.nn.neurons.format.features.DimensionScope;
import org.ml4j.nn.neurons.format.features.FeaturesFormat;

public class ML4JTensor extends TensorBase {

	private final DirectedComponentsContext directedComponentsContext;
	private final TensorDataConverter<ML4JTensorOperations> tensorDataConverter;
	private boolean isCostFunctionGradient;

	public ML4JTensor(Torch torch, DirectedComponentsContext directedComponentsContext,
			TensorDataConverter<ML4JTensorOperations> tensorDataConverter, 
			TensorData tensorData, boolean requires_grad) {
		super(torch, tensorDataConverter, tensorData);
		this.tensorDataConverter = tensorDataConverter;
		this.directedComponentsContext = directedComponentsContext;
		requires_grad_(requires_grad);
	}

	

	public ML4JTensor(Torch torch, DirectedComponentsContext directedComponentsContext,
			TensorDataConverter<ML4JTensorOperations> tensorDataConverter, 
			NeuronsActivation neuronsActivation, boolean requires_grad) {
		super(torch, tensorDataConverter,
				tensorDataConverter.createTensorDataFromTensorOperations(
						new ML4JTensorOperationsImpl(torch, directedComponentsContext,
								getMatrix(
										neuronsActivation.getActivations(directedComponentsContext.getMatrixFactory()),
										neuronsActivation),
								getSize(torch, neuronsActivation))));
		this.tensorDataConverter = tensorDataConverter;
		this.directedComponentsContext = directedComponentsContext;
		requires_grad_(requires_grad);
	}

	private static Matrix getMatrix(Matrix matrix, NeuronsActivation neuronsActivation) {
		return matrix;
	}

	public Tensor setCostFunctionGradient(boolean isCostFunctionGradient) {
		this.isCostFunctionGradient = isCostFunctionGradient;
		return this;
	}

	private static Size getSize(Torch torch, NeuronsActivation neuronsActivation) {
		return NeuronsActivationSize.getSize(torch, neuronsActivation);
	}

	public ML4JTensor(Torch torch, DirectedComponentsContext directedComponentsContext,
			TensorDataConverter<ML4JTensorOperations> tensorDataConverter, SymbolicTensor<TensorData, Size> symbolicTensor) {
		super(torch, tensorDataConverter, symbolicTensor);
		this.tensorDataConverter = tensorDataConverter;
		this.directedComponentsContext = directedComponentsContext;
	}

	public DirectedComponentsContext getDirectedComponentsContext(boolean requires_grad) {
		if (requires_grad) {
			return directedComponentsContext.asTrainingContext();
		} else {
			return directedComponentsContext.asNonTrainingContext();
		}
	}

	@Override
	protected Tensor createDefaultTensor(Torch torch, SymbolicTensor<TensorData, Size> tensor) {
		if (tensor.size().dimensions().length == 0) {
			return new ScalarImpl(torch, tensorDataConverter, tensor);
		} else {
			return new ML4JTensor(torch, directedComponentsContext, tensorDataConverter, tensor);
		}
	}

	public DirectedComponentsContext getDirectedComponentsContext() {
		return getDirectedComponentsContext(requires_grad);
	}

	public NeuronsActivation toNeuronsActivation(DimensionScope dimensionScope,
			NeuronsActivationFeatureOrientation target) {
		
		if (size().dimensionNames() == null) {
			throw new IllegalStateException("Only able to determine NeuronsActivationFormat if dimension names have been provided");
		}

		NeuronsActivationFormat<?> format = NeuronsActivationSize.getNeuronsActivationFormat(size().dimensionNames().asList(), dimensionScope);

		boolean needToTranspose = false;

		if (target != null && format.getFeatureOrientation() != target) {
			// .out.println("TRANSPOSING");
			needToTranspose = true;
		}

		ML4JTensorOperations tensorOperations = tensorDataConverter
				.createTensorOperationsFromTensorData(toTensorData());

		Integer depth = null;
		Integer width = null;
		Integer height = null;

		for (int i = 0; i < tensorOperations.size().dimensionNames().length(); i++) {
			Integer v = tensorOperations.size().get(i);
			String s = tensorOperations.size().dimensionNames().get(i);
			if (s.contains("depth")) {
				depth = v;
			}
			if (s.contains("width")) {
				width = v;
			}
			if (s.contains("height")) {
				height = v;
			}
		}

		Matrix matrix = tensorOperations.getMatrix();
		Neurons neurons = new Neurons(matrix.getColumns(), false);
		if (depth != null && width != null && height != null) {
			neurons = new Neurons3D(width, height, depth, false);
		}

		NeuronsActivationFeatureOrientation ot = target;

		NeuronsActivationFormat<?> transposedFormat = needToTranspose
				? new NeuronsActivationFormat<FeaturesFormat>(ot, format.getFeaturesFormat(),
						format.getExampleDimensions())
				: format;

		return new NeuronsActivationImpl(neurons, needToTranspose ? matrix.transpose() : matrix, transposedFormat,
				true);
	}

	public boolean isCostFunctionGradient() {
		return isCostFunctionGradient;
	}

}
