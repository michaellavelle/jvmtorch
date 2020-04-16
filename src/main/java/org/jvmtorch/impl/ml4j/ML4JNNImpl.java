package org.jvmtorch.impl.ml4j;

import org.jvmtorch.impl.NNImpl;
import org.jvmtorch.nn.Conv2d;
import org.jvmtorch.nn.Linear;
import org.jvmtorch.nn.functional.Functional;
import org.jvmtorch.nn.modules.MSELoss;
import org.jvmtorch.torch.TensorConverter;
import org.jvmtorch.torch.TensorDataConverter;
import org.jvmtorch.torch.Torch;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.factories.DirectedComponentFactory;

public class ML4JNNImpl extends NNImpl{

	private DirectedComponentFactory directedComponentFactory;
	private DirectedComponentsContext directedComponentsContext;
	private TensorDataConverter<ML4JTensorOperations> tensorDataConverter;
	private TensorConverter<ML4JTensor> tensorConverter;

	
    public ML4JNNImpl(DirectedComponentsContext directeComponentsContext, DirectedComponentFactory directedComponentFactory, Torch torch, Functional functional,
    		TensorDataConverter<ML4JTensorOperations> tensorDataConverter, TensorConverter<ML4JTensor> tensorConverter) {
        super(torch, functional);
        this.directedComponentFactory = directedComponentFactory;
        this.directedComponentsContext = directeComponentsContext;
        this.tensorDataConverter = tensorDataConverter;
        this.tensorConverter = tensorConverter;
    }

    @Override
    public Conv2d<?> Conv2d(int... params) {
        return new ML4JConv2d(this, directedComponentsContext, tensorDataConverter, tensorConverter, params[0], params[1], params[2]);
    }

    public DirectedComponentFactory getDirectedComponentFactory() {
        return directedComponentFactory;
    }

    @Override
    public Linear<?> Linear(int... params) {
        return new ML4JLinear(this, params[0], params[1]);
    }

    @Override
    public MSELoss MSELoss() {
        return new ML4JMSELossImpl(torch, tensorDataConverter, tensorConverter);
    }
}
