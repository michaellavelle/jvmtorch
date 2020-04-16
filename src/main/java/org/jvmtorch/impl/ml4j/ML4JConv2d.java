package org.jvmtorch.impl.ml4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

import org.jvmtorch.impl.TensorOperationImpl;
import org.jvmtorch.nn.Conv2d;
import org.jvmtorch.torch.Size;
import org.jvmtorch.torch.Tensor;
import org.jvmtorch.torch.TensorConverter;
import org.jvmtorch.torch.TensorData;
import org.jvmtorch.torch.TensorDataConverter;
import org.ml4j.Matrix;
import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.axons.AxonsGradient;
import org.ml4j.nn.axons.BiasVectorImpl;
import org.ml4j.nn.axons.ConvolutionalAxonsConfig;
import org.ml4j.nn.axons.FeaturesVectorFormat;
import org.ml4j.nn.axons.WeightsFormatImpl;
import org.ml4j.nn.axons.WeightsMatrixImpl;
import org.ml4j.nn.axons.WeightsMatrixOrientation;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.axons.DirectedAxonsComponent;
import org.ml4j.nn.components.axons.DirectedAxonsComponentActivation;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.neurons.ImageNeuronsActivationImpl;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.ml4j.nn.neurons.format.ImageNeuronsActivationFormat;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.ml4j.nn.neurons.format.features.Dimension;
import org.ml4j.nn.neurons.format.features.DimensionScope;

public class ML4JConv2d extends Conv2d<ML4JConv2d> {

    private DirectedComponentFactory directedComponentFactory;
    private TensorDataConverter<ML4JTensorOperations> tensorDataConverter;
    private DirectedComponentsContext directedComponentsContext;
    private TensorConverter<ML4JTensor> tensorConverter;

    public ML4JConv2d(ML4JNNImpl nn, DirectedComponentsContext directedComponentsContext, TensorDataConverter<ML4JTensorOperations> tensorDataConverter, 
    		TensorConverter<ML4JTensor> tensorConverter, int in_channels,
                      int out_channels,
                      int kernel_size) {
        super(nn, in_channels, out_channels, kernel_size);
        self.directedComponentFactory = nn.getDirectedComponentFactory();
        this.tensorDataConverter = tensorDataConverter;
        this.directedComponentsContext = directedComponentsContext;
        this.tensorConverter = 	tensorConverter;
    }
   
    @Override
    public Tensor forward(Tensor input) {
    	
		
		ML4JTensor ml4jTensor = tensorConverter.createTensor(input);
		
		NeuronsActivation inputNeuronsActivation = ml4jTensor.toNeuronsActivation(DimensionScope.INPUT, NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
		
		
		Size si = NeuronsActivationSize.getSize(torch, inputNeuronsActivation);
		
		NeuronsActivationFeatureOrientation originalFormat = inputNeuronsActivation.getFormat().getFeatureOrientation();
		boolean transposed = false;
		if (si.asList().equals(input.size().asList())) {
			transposed = true;
			if (originalFormat == NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
				originalFormat = NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET;
			} else {
				originalFormat = NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET;

			}
		}
		
    	Matrix inputMatrix = inputNeuronsActivation.getActivations(ml4jTensor.getDirectedComponentsContext().getMatrixFactory());
    	   	
    	NeuronsActivationFormat<?> format =  inputNeuronsActivation.getFormat();
    
    	boolean isCorrectFormat = ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT.isEquivalentFormat(format, DimensionScope.INPUT);

    	if (!isCorrectFormat) {
    		throw new IllegalArgumentException();
    	}
    
    
        int inputWidth = (int)Math.sqrt(inputNeuronsActivation.getFeatureCount() / self.in_channels);

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
                .createConvolutionalAxonsComponent("conv", config, new WeightsMatrixImpl(tensorDataConverter.createTensorOperationsFromTensorData(self.weight.toTensorData()).getMatrix(), new WeightsFormatImpl(Arrays.asList(Dimension.INPUT_DEPTH, Dimension.FILTER_HEIGHT, Dimension.FILTER_WIDTH),
                        Arrays.asList(Dimension.OUTPUT_DEPTH), WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS)), new BiasVectorImpl(tensorDataConverter.createTensorOperationsFromTensorData(self.bias.toTensorData()).getMatrix(), FeaturesVectorFormat.DEFAULT_BIAS_FORMAT));

        NeuronsActivation neuronsActivation
                = new ImageNeuronsActivationImpl(inputMatrix, config.getAxonsConfig().getLeftNeurons(),
                 ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT, true);
        
        
        DirectedAxonsComponentActivation activation =  convolutionalAxons.forwardPropagate(neuronsActivation, ml4jTensor.getDirectedComponentsContext());

        NeuronsActivation outActivation = activation.getOutput();

		ML4JTensor convOutput =  new ML4JTensor(torch, ml4jTensor.getDirectedComponentsContext(), tensorDataConverter,
        		"out4", "out5", outActivation, input.requires_grad());
		
		boolean transp = transposed;
		
		NeuronsActivationFeatureOrientation origFormat = originalFormat;
        		
        Tensor output = input.performUnaryMappingOperation("ConvOutput", new TensorOperationImpl<>(torch, "ConvOutput", l -> convOutput.toTensorData(), s -> convOutput.size()), new TensorOperationImpl<>(torch, "ConvBackward", l-> backward(activation, axonsConfig.getRightNeurons(), l, config.getAxonsConfig().getLeftNeurons().hasBiasUnit(), input.requires_grad(), transp, origFormat), s -> input.size()));
        
        return output;
    }
    
    
    private List<String> toScopeIndependentNamesList(List<String> strings) {
		List<String> returnValues = new ArrayList<>();
		for (String s : strings) {
			s = s.replaceAll("input_", "");
			s = s.replaceAll("output_", "");
			returnValues.add(s);
		}
		
		return returnValues;
	}
  
    private Tensor backward(DirectedAxonsComponentActivation act, Neurons3D neurons, Tensor back, boolean hasBias, boolean requires_grad, boolean transp, NeuronsActivationFeatureOrientation originalFormat) {
    	    	    	
    	if (!toScopeIndependentNamesList(back.size().dimensionNames().asList()).equals(Arrays.asList("depth", "height", "width", "example"))) {
    		throw new IllegalArgumentException("Conv backward input incorrect format");
    	}
    	
		ML4JTensor ml4jTensor = tensorConverter.createTensor(back);
		
		NeuronsActivation inputNeuronsActivation = ml4jTensor.toNeuronsActivation(DimensionScope.OUTPUT, NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
		
		
		
		
		Size si = NeuronsActivationSize.getSize(torch, inputNeuronsActivation);
		
		
		NeuronsActivationFeatureOrientation originalBackFormat = inputNeuronsActivation.getFormat().getFeatureOrientation();

		boolean backTransposed = false;
		if (si.asList().equals(back.size().asList())) {
			backTransposed = true;
			if (originalBackFormat == NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
				originalBackFormat = NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET;
			} else {
				originalBackFormat = NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET;

			}
		}
		
		
		Matrix backMatrix = inputNeuronsActivation.getActivations(ml4jTensor.getDirectedComponentsContext().getMatrixFactory());
    	
        NeuronsActivation neuronsActivation
                = new NeuronsActivationImpl(neurons, backMatrix,
                ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT, false);

        DirectedComponentGradient<NeuronsActivation> a = new
                DirectedComponentGradientImpl<>(neuronsActivation);
        DirectedComponentGradient<NeuronsActivation> outGradient =  act.backPropagate(a);
        NeuronsActivation outActivation = outGradient.getOutput();

        Supplier<AxonsGradient> b = outGradient.getTotalTrainableAxonsGradients().get(0);

        AxonsGradient axonsGradient = b.get();
        Matrix weightsGradient = axonsGradient.getWeightsGradient();
        Matrix biasGradient = axonsGradient.getLeftToRightBiasGradient();

        Tensor weightsGradientTensor = new ML4JTensor(torch, directedComponentsContext, tensorDataConverter, "weightsGrad", "weightsGrad", createTensorData(weightsGradient, self.weight.size()), false );
        self.weight.grad_(weightsGradientTensor);

        if (hasBias) {
            Tensor biasGradientTensor = new ML4JTensor(torch, directedComponentsContext, tensorDataConverter, "biasGrad", "biasGrad", createTensorData(biasGradient, self.bias.size()), false);
            self().bias.grad_(biasGradientTensor);
        }

        ML4JTensor output =  new ML4JTensor(torch, ml4jTensor.getDirectedComponentsContext(), tensorDataConverter, "out10", "out11", outActivation, true);
        
        if (backTransposed && !transp) {
    		NeuronsActivation outAct = output.toNeuronsActivation(DimensionScope.OUTPUT, originalBackFormat);
    		output =  new ML4JTensor(torch, ml4jTensor.getDirectedComponentsContext(), tensorDataConverter, "out12", "out13", outAct, true);
        }
        


        return output;
    }
    
    private TensorData createTensorData(Matrix matrix, Size size) {
		return tensorDataConverter.createTensorDataFromTensorOperations(new ML4JTensorOperationsImpl(torch, directedComponentsContext, matrix, size));
	}

    @Override
    public ML4JConv2d self() {
        return this;
    }
}
