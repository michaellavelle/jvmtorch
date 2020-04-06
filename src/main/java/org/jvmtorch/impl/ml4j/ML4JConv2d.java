package org.jvmtorch.impl.ml4j;

import org.jvmtorch.impl.TensorOperationImpl;
import org.jvmtorch.nn.Conv2d;
import org.jvmtorch.torch.Tensor;
import org.ml4j.Matrix;
import org.ml4j.nn.axons.*;
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
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.ml4j.nn.neurons.format.ImageNeuronsActivationFormat;
import org.ml4j.nn.neurons.format.features.Dimension;

import java.util.Arrays;
import java.util.function.Supplier;

public class ML4JConv2d extends Conv2d<ML4JConv2d, ML4JTensorOperations> {

    private DirectedComponentsContext directedComponentsContext;
    private DirectedComponentFactory directedComponentFactory;

    public ML4JConv2d(ML4JNNImpl nn, int in_channels,
                      int out_channels,
                      int kernel_size) {
        super(nn, in_channels, out_channels, kernel_size);
        self.directedComponentFactory = nn.getDirectedComponentFactory();
        self.directedComponentsContext = new DirectedComponentsContextImpl(nn.getMatrixFactory(), true);
    }

    @Override
    public Tensor<ML4JTensorOperations> forward(Tensor<ML4JTensorOperations> input) {
        Matrix inputMatrix = input.toTensorOperations().getMatrix();
        inputMatrix.setImmutable(true);
        int inputWidth = (int)Math.sqrt(inputMatrix.getColumns() / self.in_channels);

        int inputHeight = inputWidth;
        int outputWidth = inputWidth - self.kernel_size + 1;
        int outputHeight = outputWidth;

        Axons3DConfig axonsConfig = new Axons3DConfig(new
                Neurons3D(inputWidth, inputHeight, self.in_channels, true),
                new Neurons3D(outputWidth, outputHeight, self.out_channels, false))
                .withFilterHeight(self.kernel_size)
                .withFilterWidth(self.kernel_size);
        ConvolutionalAxonsConfig config = new ConvolutionalAxonsConfig(axonsConfig);

        DirectedAxonsComponent<Neurons3D, Neurons3D, ?> convolutionalAxons
                = directedComponentFactory
                .createConvolutionalAxonsComponent("conv", config, new WeightsMatrixImpl(self.weight.toTensorOperations().getMatrix(), new WeightsFormatImpl(Arrays.asList(Dimension.INPUT_DEPTH, Dimension.FILTER_HEIGHT, Dimension.FILTER_WIDTH),
                        Arrays.asList(Dimension.OUTPUT_DEPTH), WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS)), new BiasVectorImpl(self.bias.toTensorOperations().getMatrix(), FeaturesVectorFormat.DEFAULT_BIAS_FORMAT));

        NeuronsActivation neuronsActivation
                = new ImageNeuronsActivationImpl(inputMatrix.transpose(), config.getAxonsConfig().getLeftNeurons(),
                 ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT, true);

        DirectedAxonsComponentActivation activation =  convolutionalAxons.forwardPropagate(neuronsActivation, directedComponentsContext);

        Matrix outMatrix = activation.getOutput().getActivations(directedComponentsContext.getMatrixFactory());

        Tensor<ML4JTensorOperations> convOutput =  new ML4JTensor(torch, directedComponentsContext.getMatrixFactory(), "out", "out", outMatrix.transpose());
        convOutput.requires_grad_(true);

        Tensor<ML4JTensorOperations> output = input.performUnaryMappingOperation("ConvOutput", new TensorOperationImpl<>("ConvOutput", l -> convOutput.toTensorOperations(), convOutput.size()), new TensorOperationImpl<>("ConvBackward", l-> backward(activation, axonsConfig.getRightNeurons(), l, config.getAxonsConfig().getLeftNeurons().hasBiasUnit()), input.size()));

        return output;
    }

    private Tensor<ML4JTensorOperations> backward(DirectedAxonsComponentActivation act, Neurons3D neurons, Tensor<ML4JTensorOperations> back, boolean hasBias) {

        NeuronsActivation neuronsActivation
                = new NeuronsActivationImpl(neurons, back.toTensorOperations().getMatrix().transpose(),
                ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT, false);

        DirectedComponentGradient<NeuronsActivation> a = new
                DirectedComponentGradientImpl<>(neuronsActivation);
        DirectedComponentGradient<NeuronsActivation> outGradient =  act.backPropagate(a);
        Matrix outMatrix = outGradient.getOutput().getActivations(directedComponentsContext.getMatrixFactory());

        Supplier<AxonsGradient> b = outGradient.getTotalTrainableAxonsGradients().get(0);

        AxonsGradient axonsGradient = b.get();
        Matrix weightsGradient = axonsGradient.getWeightsGradient();
        Matrix biasGradient = axonsGradient.getLeftToRightBiasGradient();
        Tensor<ML4JTensorOperations> weightsGradientTensor = new ML4JTensor(torch, directedComponentsContext.getMatrixFactory(), "weightsGrad", "weightsGrad", weightsGradient);
        self.weight.grad_(weightsGradientTensor);

        if (hasBias) {
            Tensor<ML4JTensorOperations> biasGradientTensor = new ML4JTensor(torch, directedComponentsContext.getMatrixFactory(), "weightsGrad", "weightsGrad", biasGradient);

            self().bias.grad_(biasGradientTensor);
        }

        Tensor<ML4JTensorOperations> output =  new ML4JTensor(torch, directedComponentsContext.getMatrixFactory(), "out", "out", outMatrix.transpose()).requires_grad_(true);
        return output;
    }

    @Override
    public ML4JConv2d self() {
        return this;
    }
}
