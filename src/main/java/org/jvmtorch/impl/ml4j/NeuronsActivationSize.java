package org.jvmtorch.impl.ml4j;

import static org.jvmpy.python.Python.tuple;

import java.util.ArrayList;
import java.util.List;

import org.jvmtorch.torch.Size;
import org.jvmtorch.torch.Torch;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.format.features.Dimension;

public class NeuronsActivationSize {

	public static Size getSize(Torch torch, NeuronsActivation neuronsActivation) {
		
		if (neuronsActivation.getNeurons() instanceof Neurons3D) {
					
			Neurons3D neurons = (Neurons3D)neuronsActivation.getNeurons();
			int[] vals = new int[neuronsActivation.getFormat().getFeaturesFormat().getDimensions().size()];
			List<String> names = new ArrayList<>();
			int index = 0;
			for (Dimension dim : neuronsActivation.getFormat().getFeaturesFormat().getDimensions()) {
				vals[index] = getVal(neurons, neuronsActivation.getExampleCount(), dim);
				names.add(getName(neurons, dim));
				index++;
			}
				
			if (neuronsActivation.getFeatureOrientation() == NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
				names.add("example");
				return torch.Size(torch.Size(vals),  torch.Size(neuronsActivation.getExampleCount())).names_(tuple(names));
			} else {
				names.add(0, "example");
				return torch.Size(torch.Size(neuronsActivation.getExampleCount()),  torch.Size(vals)).names_(tuple(names));
			}
		
		} else {
			neuronsActivation.getFormat().getDimensions();
			
			if (neuronsActivation.getFeatureOrientation() == NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
			
				return torch.Size(neuronsActivation.getFeatureCount(),  neuronsActivation.getExampleCount()).names_(tuple("feature", "example"));
			} else {
				return torch.Size(neuronsActivation.getExampleCount(),  neuronsActivation.getFeatureCount()).names_(tuple("example", "feature"));
			}
		}
		
	}
	
	private static String getName(Neurons3D neurons, Dimension dim) {
		if (dim == Dimension.INPUT_DEPTH) {
			return "input_depth";
		} else if (dim == Dimension.DEPTH) {
			return "depth";
		} else if (dim == Dimension.OUTPUT_DEPTH) {
			return "output_depth";
		} else if (dim == Dimension.INPUT_WIDTH) {
			return "input_width";
		} else if (dim == Dimension.WIDTH) {
			return "width";
		} else if (dim == Dimension.OUTPUT_WIDTH) {
			return "output_width";
		} else if (dim == Dimension.INPUT_HEIGHT) {
			return "input_height";
		} else if (dim == Dimension.HEIGHT) {
			return "height";
		} else if (dim == Dimension.OUTPUT_HEIGHT) {
			return "output_height";
		} else if (dim == Dimension.EXAMPLE) {
			return "example";
		} else {
			throw new IllegalArgumentException();
		}
	}

	private static int getVal(Neurons3D neurons, int examples, Dimension dim) {
		if (dim == Dimension.INPUT_DEPTH) {
			return neurons.getDepth();
		} else if (dim == Dimension.DEPTH) {
			return neurons.getDepth();
		} else if (dim == Dimension.OUTPUT_DEPTH) {
			return neurons.getDepth();
		} else if (dim == Dimension.INPUT_WIDTH) {
			return neurons.getWidth();
		} else if (dim == Dimension.WIDTH) {
			return neurons.getWidth();
		} else if (dim == Dimension.OUTPUT_WIDTH) {
			return neurons.getWidth();
		} else if (dim == Dimension.INPUT_HEIGHT) {
			return neurons.getHeight();
		} else if (dim == Dimension.HEIGHT) {
			return neurons.getHeight();
		} else if (dim == Dimension.OUTPUT_HEIGHT) {
			return neurons.getHeight();
		} else if (dim == Dimension.EXAMPLE) {
			return examples;
		} else {
			throw new IllegalArgumentException();
		}
	}
}
