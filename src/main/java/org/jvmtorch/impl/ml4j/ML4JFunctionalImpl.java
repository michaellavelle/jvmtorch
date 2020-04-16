package org.jvmtorch.impl.ml4j;

import static org.jvmtorch.JvmTorch.torch;

import org.jvmtorch.impl.FunctionalImpl;
import org.jvmtorch.impl.TensorOperationImpl;
import org.jvmtorch.nn.functional.Functional;
import org.jvmtorch.torch.Size;
import org.jvmtorch.torch.Tensor;
import org.jvmtorch.torch.TensorConverter;
import org.jvmtorch.torch.TensorDataConverter;
import org.ml4j.Matrix;
import org.ml4j.nn.activationfunctions.ActivationFunctionBaseType;
import org.ml4j.nn.activationfunctions.ActivationFunctionProperties;
import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.axons.PoolingAxonsConfig;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponent;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponentActivation;
import org.ml4j.nn.components.axons.DirectedAxonsComponent;
import org.ml4j.nn.components.axons.DirectedAxonsComponentActivation;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.neurons.ImageNeuronsActivationImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.format.ImageNeuronsActivationFormat;
import org.ml4j.nn.neurons.format.features.DimensionScope;

public class ML4JFunctionalImpl extends FunctionalImpl implements Functional {

	private DirectedComponentFactory directedComponentFactory;
	private TensorDataConverter<ML4JTensorOperations> tensorDataConverter;
	private TensorConverter<ML4JTensor> tensorConverter;

	public ML4JFunctionalImpl(DirectedComponentFactory directedComponentFactory,
			TensorDataConverter<ML4JTensorOperations> tensorDataConverter,
			TensorConverter<ML4JTensor> tensorConverter) {
		this.directedComponentFactory = directedComponentFactory;
		this.tensorDataConverter = tensorDataConverter;
		this.tensorConverter = tensorConverter;
	}

	public Tensor activationFunction(Tensor input, ActivationFunctionType activationFunctionType, boolean isCostFunctionGradient) {

		ML4JTensor ml4jTensor = tensorConverter.createTensor(input);
		
		NeuronsActivationFeatureOrientation target = null;
		if (activationFunctionType.getBaseType() == ActivationFunctionBaseType.SOFTMAX) {
			target = NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET;
		}
		
		NeuronsActivation neuronsActivation = ml4jTensor.toNeuronsActivation(DimensionScope.INPUT, target);		
		
		final DifferentiableActivationFunctionComponent activationFunction = directedComponentFactory
				.createDifferentiableActivationFunctionComponent(activationFunctionType.toString(),
						neuronsActivation.getNeurons(), activationFunctionType, new ActivationFunctionProperties());
		
		DifferentiableActivationFunctionComponentActivation act = activationFunction.forwardPropagate(neuronsActivation,
				ml4jTensor.getDirectedComponentsContext());

		NeuronsActivation outActivation = act.getOutput();
			
						
		ML4JTensor outTensor = new ML4JTensor(torch, ml4jTensor.getDirectedComponentsContext(), tensorDataConverter, "out",
				"out", outActivation, input.requires_grad());
					
		Tensor output = input.performUnaryMappingOperation(activationFunctionType.toString() + "Output",
				new TensorOperationImpl<>(torch, activationFunctionType.toString(), l -> outTensor.toTensorData(),
						s -> outTensor.size()),
				new TensorOperationImpl<>(torch, activationFunctionType.toString() + "Backward",
						l -> backward(activationFunctionType, act, outActivation.getNeurons(), l, input.requires_grad()), s -> input.size()));

		return output;
	}

	@Override
	public Tensor relu(Tensor input) {

		return activationFunction(input, ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), false);
	}

	private Tensor backward(ActivationFunctionType activationFunctionType, DifferentiableActivationFunctionComponentActivation act, Neurons neurons, Tensor back,
			boolean requires_grad) {
		
		ML4JTensor ml4jTensor = tensorConverter.createTensor(back);
		
		NeuronsActivationFeatureOrientation target = null;
		if (activationFunctionType.getBaseType() == ActivationFunctionBaseType.SOFTMAX) {
			target = NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET;
		}
		
		NeuronsActivation neuronsActivation = ml4jTensor.toNeuronsActivation(DimensionScope.OUTPUT, target);
		
		DirectedComponentsContext directedComponentsContext = ml4jTensor.getDirectedComponentsContext();

		DirectedComponentGradient<NeuronsActivation> a = new DirectedComponentGradientImpl<>(neuronsActivation);
		
		NeuronsActivation outNeuronsActivation = neuronsActivation;
		if (!ml4jTensor.isCostFunctionGradient()) {
			outNeuronsActivation = act.backPropagate(a).getOutput();	
		} 
		
		Tensor output = new ML4JTensor(torch, directedComponentsContext, tensorDataConverter, "out", "out",
				outNeuronsActivation, requires_grad);
		
		return output;

	}

	@Override
	public Tensor softmax(Tensor input) {

		boolean isCostFunctionGradient = false;
		
		if (input instanceof ML4JTensor) {
			isCostFunctionGradient = ((ML4JTensor)input).isCostFunctionGradient();
		}
		return activationFunction(input, 
				ActivationFunctionType.getBaseType(ActivationFunctionBaseType.SOFTMAX), isCostFunctionGradient);

	}
	
	@Override
	public Tensor sigmoid(Tensor input) {
		
		boolean isCostFunctionGradient = false;
		
		if (input instanceof ML4JTensor) {
			isCostFunctionGradient = ((ML4JTensor)input).isCostFunctionGradient();
		}

		return activationFunction(input, ActivationFunctionType.getBaseType(ActivationFunctionBaseType.SIGMOID), isCostFunctionGradient);

	}

	@Override
	public Tensor max_pool2d(Tensor input, Size size) {
		
		ML4JTensor ml4jTensor = tensorConverter.createTensor(input);
				
		NeuronsActivation inputNeuronsActivation = ml4jTensor.toNeuronsActivation(DimensionScope.INPUT, NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
		
		Matrix inputMatrix = inputNeuronsActivation.getActivations(ml4jTensor.getDirectedComponentsContext().getMatrixFactory());


		PoolingAxonsConfig config = getMaxPoolConfig(inputNeuronsActivation, size);

		DirectedAxonsComponent<Neurons3D, Neurons3D, ?> maxPoolingAxons = directedComponentFactory
				.createMaxPoolingAxonsComponent("maxpool", config, true);
		
		NeuronsActivation neuronsActivation = new ImageNeuronsActivationImpl(inputMatrix,
				config.getAxonsConfig().getLeftNeurons(), ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT, true);

		DirectedAxonsComponentActivation activation = maxPoolingAxons.forwardPropagate(neuronsActivation,
				ml4jTensor.getDirectedComponentsContext());
		
		NeuronsActivation outNeuronsActivation= activation.getOutput();

		Tensor convOutput = new ML4JTensor(torch, ml4jTensor.getDirectedComponentsContext(), tensorDataConverter,
				"maxpoolout", "maxpoolout2", outNeuronsActivation, input.requires_grad());
		
		convOutput.requires_grad_(true);
		Tensor output = input.performUnaryMappingOperation("ConvOutput",
				new TensorOperationImpl<>(torch, "ConvOutput", l -> convOutput.toTensorData(), s -> convOutput.size()),
				new TensorOperationImpl<>(torch, "ConvBackward",
						l -> backward(activation, config.getAxonsConfig().getRightNeurons(), l, input.requires_grad()), s -> input.size()));

		return output;

	}
	
	private PoolingAxonsConfig getMaxPoolConfig(NeuronsActivation inputNeuronsActivation, Size pool_size) {
		Integer inputWidth = null;
		Integer inputHeight = null;
		Integer inputDepth = null;

		if (inputNeuronsActivation.getNeurons() instanceof Neurons3D) {
			Neurons3D neurons3D = (Neurons3D)inputNeuronsActivation.getNeurons();
			inputWidth = neurons3D.getWidth();
			inputHeight = neurons3D.getHeight();
			inputDepth = neurons3D.getDepth();
			
		} else {
			throw new IllegalArgumentException("");
		}
	
		int filterHeight = pool_size.get(0);
		int filterWidth = pool_size.numel() == 1 ? pool_size.get(0) : pool_size.get(1);
		int strideWidth = filterWidth;
		int strideHeight= filterHeight;

		int outputWidth=(inputWidth-filterWidth)/strideWidth +1;
		int outputHeight=(inputHeight-filterHeight)/strideHeight +1;

		Axons3DConfig axonsConfig = new Axons3DConfig(new Neurons3D(inputWidth, inputHeight, inputDepth, false),
				new Neurons3D(outputWidth, outputHeight, inputDepth, false)).withStrideWidth(strideWidth).withStrideHeight(strideHeight);
		PoolingAxonsConfig config = new PoolingAxonsConfig(axonsConfig);
		return config;
	}

	@Override
	public Tensor max_pool2d(Tensor input, int pool_size) {
		return max_pool2d(input, torch.Size(pool_size, pool_size));

	}

	private Tensor backward(DirectedAxonsComponentActivation act, Neurons3D neurons, Tensor back,
			boolean requires_grad) {	
		
		ML4JTensor ml4jTensor = tensorConverter.createTensor(back);
		
		NeuronsActivation inputNeuronsActivation = ml4jTensor.toNeuronsActivation(DimensionScope.OUTPUT, NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);

		Matrix backMatrix = inputNeuronsActivation.getActivations(ml4jTensor.getDirectedComponentsContext().getMatrixFactory());

		NeuronsActivation neuronsActivation = new ImageNeuronsActivationImpl(backMatrix, neurons,
				ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT, false);

		DirectedComponentGradient<NeuronsActivation> gradient = new DirectedComponentGradientImpl<>(neuronsActivation);
		
		NeuronsActivation out = act.backPropagate(gradient).getOutput();
		
		Tensor output = new ML4JTensor(torch, ml4jTensor.getDirectedComponentsContext(), 
				tensorDataConverter, "actback", "actback1",
				out, requires_grad);
		
		return output;
	}


}
