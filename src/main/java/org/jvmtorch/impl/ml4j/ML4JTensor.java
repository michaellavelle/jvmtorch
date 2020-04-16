package org.jvmtorch.impl.ml4j;

import java.util.ArrayList;
import java.util.List;

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
import org.ml4j.nn.neurons.format.features.Dimension;
import org.ml4j.nn.neurons.format.features.DimensionScope;
import org.ml4j.nn.neurons.format.features.FeaturesFormat;

public class ML4JTensor extends TensorBase {
	
	private final DirectedComponentsContext directedComponentsContext;
	private final TensorDataConverter<ML4JTensorOperations> tensorDataConverter;
	private boolean isCostFunctionGradient;

	public ML4JTensor(Torch torch, DirectedComponentsContext directedComponentsContext, TensorDataConverter<ML4JTensorOperations> tensorDataConverter, String name, String inputName,
			TensorData tensorData, boolean requires_grad) {
		super(torch, tensorDataConverter, name, inputName, tensorData);
		this.tensorDataConverter = tensorDataConverter;
		this.directedComponentsContext = directedComponentsContext;
		requires_grad_(requires_grad);
	}
	
	public NeuronsActivationFormat<?> getNeuronsActivationFormat(DimensionScope dimensionScope) {
    	    	
    	List<Dimension> nonExampleDimensions = new ArrayList<>();
    	List<Dimension> exampleDimensions = new ArrayList<>();
    	List<Dimension> allDimensions = new ArrayList<>();


    	for (String name : size().dimensionNames()) {
    		String n = name.substring(0, 1).toUpperCase() + name.substring(1);
    		String id = name.substring(0, 1).toUpperCase();
    		int ind = n.indexOf("_");
    		int prevInd = 0;
    		String n2 = n;
    		while (ind != -1) {
        		n2 = n2.substring(prevInd, ind + 1) +  n2.substring(ind + 1, ind + 2).toUpperCase() + n2.substring(ind + 2);
        		id = id + name.substring(ind + 1, ind + 2).toUpperCase();
        		prevInd = ind;
        		ind = n2.indexOf("_", prevInd + 1);
    		}
    		
    		Dimension dimension = new Dimension(id, n2, dimensionScope);
    		boolean exampleDimension = false;
    		
    		if (dimension.getId().equals(Dimension.INPUT_DEPTH.getId())) {
    			dimension = Dimension.INPUT_DEPTH;
    		} else if (dimension.getId().equals(Dimension.INPUT_WIDTH.getId())) {
    			dimension = Dimension.INPUT_WIDTH;
    		}
    		else if (dimension.getId().equals(Dimension.INPUT_HEIGHT.getId())) {
    			dimension = Dimension.INPUT_HEIGHT;
    		}
    		else if (dimension.getId().equals(Dimension.EXAMPLE.getId())) {
    			dimension = Dimension.EXAMPLE;
    			exampleDimension = true;
    		}
    		
    		else if (dimension.getId().equals(Dimension.OUTPUT_DEPTH.getId())) {
    			dimension = Dimension.OUTPUT_DEPTH;
    		}
    		else if (dimension.getId().equals(Dimension.OUTPUT_WIDTH.getId())) {
    			dimension = Dimension.OUTPUT_WIDTH;
    		}
    		else if (dimension.getId().equals(Dimension.OUTPUT_HEIGHT.getId())) {
    			dimension = Dimension.OUTPUT_HEIGHT;
    		}
    		else if (dimension.getId().equals(Dimension.DEPTH.getId())) {
    			dimension = Dimension.DEPTH;
    		}
    		else if (dimension.getId().equals(Dimension.WIDTH.getId())) {
    			dimension = Dimension.WIDTH;
    		}
    		else if (dimension.getId().equals(Dimension.HEIGHT.getId())) {
    			dimension = Dimension.HEIGHT;
    		}
    		
    		else if (dimension.getId().equals(Dimension.FEATURE.getId())) {
    			dimension = Dimension.FEATURE;
    		}
    		
    		else if (dimension.getId().equals(Dimension.INPUT_FEATURE.getId())) {
    			dimension = Dimension.INPUT_FEATURE;
    		}
    		
    		else if (dimension.getId().equals(Dimension.OUTPUT_FEATURE.getId())) {
    			dimension = Dimension.OUTPUT_FEATURE;
    		} 
  
    		else {
    			throw new IllegalStateException("Unable to extract NeuronsActivationFormat - not all the dimensions of the Tensor have recognisable names - eg. " + dimension.getName());
    		}
    		
    		if (exampleDimension) {
    			exampleDimensions.add(dimension);
    		} else {
    			nonExampleDimensions.add(dimension);
    		}
    		allDimensions.add(dimension);
    	}
    	
    	FeaturesFormat featuresFormat = new FeaturesFormat() {

			@Override
			public List<Dimension> getDimensions() {
				return nonExampleDimensions;
			}
    		
    	};
    	
    	if (exampleDimensions.size() != 1) {
    		throw new IllegalStateException();
    	}
    	
    	
    	NeuronsActivationFeatureOrientation fo = allDimensions.get(0).equals(exampleDimensions.get(0)) ? 
    			NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET : NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET;
    	
    	NeuronsActivationFormat<?> f= new NeuronsActivationFormat<FeaturesFormat>(
    			fo, featuresFormat, exampleDimensions);
    	return f;
    }
	
	public ML4JTensor(Torch torch, DirectedComponentsContext directedComponentsContext, 
			TensorDataConverter<ML4JTensorOperations> tensorDataConverter, String name, String inputName,
			NeuronsActivation neuronsActivation, boolean requires_grad) {
		super(torch, tensorDataConverter, name, inputName, 
				tensorDataConverter.createTensorDataFromTensorOperations(
						new ML4JTensorOperationsImpl(torch, directedComponentsContext, getMatrix(neuronsActivation.getActivations(directedComponentsContext.getMatrixFactory()), neuronsActivation),
								getSize(torch, neuronsActivation))
								));
		this.tensorDataConverter = tensorDataConverter;
		this.directedComponentsContext = directedComponentsContext;
		requires_grad_(requires_grad);
	}
	
	private static Matrix getMatrix(Matrix matrix, NeuronsActivation neuronsActivation) {
		return matrix;
		//return neuronsActivation.getFeatureOrientation() == NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET ? matrix.transpose() : matrix;
	}
	
	public Tensor setCostFunctionGradient(boolean isCostFunctionGradient) {
		this.isCostFunctionGradient = isCostFunctionGradient;
		return this;
	}

	private static Size getSize(Torch torch, NeuronsActivation neuronsActivation) {
		return NeuronsActivationSize.getSize(torch, neuronsActivation);
		//return new Size(neuronsActivation.getExampleCount(), neuronsActivation.getFeatureCount());
	}

	public ML4JTensor(Torch torch, DirectedComponentsContext directedComponentsContext, TensorDataConverter<ML4JTensorOperations> tensorDataConverter,
			SymbolicTensor<TensorData> symbolicTensor) {
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
	protected Tensor createDefaultTensor(Torch torch, SymbolicTensor<TensorData> tensor) {
		if (tensor.dimensions().length == 0) {
			return new ScalarImpl(torch, tensorDataConverter, tensor);
		} else {
			return new ML4JTensor(torch, directedComponentsContext, tensorDataConverter, tensor);
		}
	}

	public DirectedComponentsContext getDirectedComponentsContext() {
		return getDirectedComponentsContext(requires_grad);
	}
	
	public NeuronsActivation toNeuronsActivation(DimensionScope dimensionScope, NeuronsActivationFeatureOrientation target) {
		
		NeuronsActivationFormat<?> format = getNeuronsActivationFormat(dimensionScope);
	
		boolean needToTranspose = false;
		
		if (target != null && format.getFeatureOrientation() != target) {
			//.out.println("TRANSPOSING");
			needToTranspose = true;
		}
		
		
		ML4JTensorOperations tensorOperations = 
				tensorDataConverter.createTensorOperationsFromTensorData(toTensorData());
		
		
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
		// TODO
		Neurons neurons = new Neurons(matrix.getColumns(), false);
		if (depth != null && width != null && height != null) {
			neurons = new Neurons3D(width, height, depth, false);
		} 
		
		NeuronsActivationFeatureOrientation ot = target;
		
		NeuronsActivationFormat<?> transposedFormat = needToTranspose ? new NeuronsActivationFormat<FeaturesFormat>(ot, format.getFeaturesFormat(), 
				format.getExampleDimensions()) : format;
		
		return  new NeuronsActivationImpl( neurons, needToTranspose ? matrix.transpose() : matrix,
				transposedFormat, true);
	}

	public boolean isCostFunctionGradient() {
		return isCostFunctionGradient;
	}

}
