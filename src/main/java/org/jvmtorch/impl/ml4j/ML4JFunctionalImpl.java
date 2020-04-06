package org.jvmtorch.impl.ml4j;

import org.jvmtorch.impl.TensorOperationImpl;
import org.jvmtorch.nn.Functional;
import org.jvmtorch.nn.Parameter;
import org.jvmtorch.torch.Tensor;
import org.jvmpy.python.Tuple;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.axons.PoolingAxonsConfig;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.DirectedComponentsContextImpl;
import org.ml4j.nn.components.axons.DirectedAxonsComponent;
import org.ml4j.nn.components.axons.DirectedAxonsComponentActivation;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.neurons.ImageNeuronsActivationImpl;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.format.ImageNeuronsActivationFormat;

import static org.jvmtorch.impl.ml4j.ML4JJvmTorch.torch;

public class ML4JFunctionalImpl implements Functional<ML4JTensorOperations> {

	private MatrixFactory matrixFactory;
	private DirectedComponentsContext directedComponentsContext;
	private DirectedComponentFactory directedComponentFactory;
	
	public ML4JFunctionalImpl(DirectedComponentFactory directedComponentFactory, MatrixFactory matrixFactory) {
		this.matrixFactory = matrixFactory;
		this.directedComponentFactory = directedComponentFactory;
		this.directedComponentsContext = new DirectedComponentsContextImpl(matrixFactory, true);
	}

	@Override
	public Tensor<ML4JTensorOperations> relu(Tensor<ML4JTensorOperations> input) {



		// TODO
		Tensor<ML4JTensorOperations> output = input.performUnaryMappingOperation("ReluOutput", new TensorOperationImpl<>("Relu", l -> l, input.size()), new TensorOperationImpl<>("ReluBackward", l->l, input.size()));

		return output;
	}

	@Override
	public Tensor<ML4JTensorOperations> max_pool2d(Tensor<ML4JTensorOperations> input, Tuple<Integer> tuple) {
		// TODO

		Matrix inputMatrix = input.toTensorOperations().getMatrix();

		int inputWidth = (int)Math.sqrt(inputMatrix.getColumns() / 6);
		int outputWidth = inputWidth/2;
		int inputHeight = inputWidth;
		int outputHeight = outputWidth;


		//System.out.println("Output max pool:" + outputHeight);

		Axons3DConfig axonsConfig = new Axons3DConfig(new
				Neurons3D(inputWidth, inputHeight, 6, false),
				new Neurons3D(outputWidth, outputHeight, 6, false))
				.withStrideWidth(2).withStrideHeight(2);
		PoolingAxonsConfig config = new PoolingAxonsConfig(axonsConfig);

		DirectedAxonsComponent<Neurons3D, Neurons3D, ?> maxPoolingAxons
				= directedComponentFactory
				.createMaxPoolingAxonsComponent("maxpool", config, false);

		NeuronsActivation neuronsActivation
				= new ImageNeuronsActivationImpl(inputMatrix.transpose(), config.getAxonsConfig().getLeftNeurons(),
				ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT, true);

		DirectedAxonsComponentActivation activation =  maxPoolingAxons.forwardPropagate(neuronsActivation, directedComponentsContext);

		Matrix outMatrix = activation.getOutput().getActivations(directedComponentsContext.getMatrixFactory());


		ML4JTensor convOutput =  new ML4JTensor(torch, directedComponentsContext.getMatrixFactory(), "out", "out", outMatrix.transpose());
		convOutput.requires_grad_(true);
		Tensor<ML4JTensorOperations> output = input.performUnaryMappingOperation("ConvOutput", new TensorOperationImpl<>("ConvOutput", l -> convOutput.toTensorOperations(), convOutput.size()), new TensorOperationImpl<>("ConvBackward", l-> backward(activation, axonsConfig.getRightNeurons(), l), input.size()));

		return output;

	}

	@Override
	public Tensor<ML4JTensorOperations> max_pool2d(Tensor<ML4JTensorOperations> input, int i) {

		Matrix inputMatrix = input.toTensorOperations().getMatrix();


		int inputWidth = (int)Math.sqrt(inputMatrix.getColumns() / 16);
		int outputWidth = inputWidth/2;
		int inputHeight = inputWidth;
		int outputHeight = outputWidth;

		Axons3DConfig axonsConfig = new Axons3DConfig(new
				Neurons3D(inputWidth, inputHeight, 16, false),
				new Neurons3D(outputWidth, outputHeight, 16, false))
				.withStrideWidth(2).withStrideHeight(2);
		PoolingAxonsConfig config = new PoolingAxonsConfig(axonsConfig);

		DirectedAxonsComponent<Neurons3D, Neurons3D, ?> maxPoolingAxons
				= directedComponentFactory
				.createMaxPoolingAxonsComponent("maxpool", config, false);

		NeuronsActivation neuronsActivation
				= new ImageNeuronsActivationImpl(inputMatrix.transpose(), config.getAxonsConfig().getLeftNeurons(),
				ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT, true);

		DirectedAxonsComponentActivation activation =  maxPoolingAxons.forwardPropagate(neuronsActivation, directedComponentsContext);

		Matrix outMatrix = activation.getOutput().getActivations(directedComponentsContext.getMatrixFactory());

		ML4JTensor convOutput =  new ML4JTensor(torch, directedComponentsContext.getMatrixFactory(), "out", "out", outMatrix.transpose());
		convOutput.requires_grad_(true);

		Tensor<ML4JTensorOperations> output = input.performUnaryMappingOperation("ConvOutput", new TensorOperationImpl<>("ConvOutput", l -> convOutput.toTensorOperations(), convOutput.size()), new TensorOperationImpl<>("ConvBackward", l-> backward(activation, axonsConfig.getRightNeurons(), l), input.size()));

		return output;

	}

	private Tensor<ML4JTensorOperations> backward(DirectedAxonsComponentActivation act, Neurons3D neurons, Tensor<ML4JTensorOperations> back) {

		Matrix backMatrix = back.toTensorOperations().getMatrix();

		NeuronsActivation neuronsActivation
				= new ImageNeuronsActivationImpl(backMatrix.transpose(), neurons,
				ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT, false);



		DirectedComponentGradient<NeuronsActivation> a = new
				DirectedComponentGradientImpl<>(neuronsActivation);
		Matrix outMatrix =  act.backPropagate(a).getOutput().getActivations(directedComponentsContext.getMatrixFactory());
		Tensor<ML4JTensorOperations> output =  new ML4JTensor(torch, directedComponentsContext.getMatrixFactory(), "out", "out", outMatrix.transpose());

		return output;
	}

	@Override
	public Tensor<ML4JTensorOperations> linear(Tensor<ML4JTensorOperations> input, Parameter<ML4JTensorOperations> weight, Parameter<ML4JTensorOperations> bias) {

		var output = input.matmul(weight.t());
		if (bias != null) {
			output = output.add(bias);
		}
		return output;
	}

}
