package org.jvmtorch.impl.ml4j;

import org.jvmtorch.torch.Tensor;
import org.jvmtorch.torch.TensorConverter;
import org.jvmtorch.torch.TensorDataConverter;
import org.ml4j.nn.components.DirectedComponentsContext;

public class ML4JTensorConverter implements TensorConverter<ML4JTensor> {

	private TensorDataConverter<ML4JTensorOperations> tensorDataConverter;
	private DirectedComponentsContext directedComponentsContext;
	
	public ML4JTensorConverter(TensorDataConverter<ML4JTensorOperations> tensorDataConverter, 
			DirectedComponentsContext directedComponentsContext) {
		this.tensorDataConverter = tensorDataConverter;
		this.directedComponentsContext = directedComponentsContext;
	}
	
	@Override
	public ML4JTensor createTensor(Tensor tensor) {
		if (tensor instanceof ML4JTensor) {
			return ML4JTensor.class.cast(tensor);
		} else {
			ML4JTensor ml4jTensor = new ML4JTensor(tensor.torch(), directedComponentsContext, tensorDataConverter, "name", "inputName", 
					tensor.toTensorData(), tensor.requires_grad());
			//ml4jTensor.size_(tensor.size());
			//if (tensor.names().asList().size() == ml4jTensor.names().asList().size()) {
				//ml4jTensor.names_(tensor.names());
			//}
			return ml4jTensor;
		}
	}
}
