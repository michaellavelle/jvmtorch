package org.jvmtorch.impl.ml4j;

import org.jvmtorch.impl.MSELossImpl;
import org.jvmtorch.impl.TensorOperationImpl;
import org.jvmtorch.nn.modules.MSELoss;
import org.jvmtorch.torch.Tensor;
import org.jvmtorch.torch.Torch;

public class ML4JMSELossImpl extends MSELossImpl<ML4JTensorOperations> {

    private Torch<ML4JTensorOperations> torch;

    public ML4JMSELossImpl(Torch<ML4JTensorOperations> torch) {
        this.torch = torch;
    }

    @Override
    public Tensor<ML4JTensorOperations> forward(MSELoss<ML4JTensorOperations> self,
                                                Tensor<ML4JTensorOperations> input,
                                                Tensor<ML4JTensorOperations> target) {
        // TODO
        Tensor<ML4JTensorOperations> output = input.performUnaryMappingOperation("LossOutput",
                new TensorOperationImpl<ML4JTensorOperations>("LossOutput", l -> torch.ones(1, 1).toTensorOperations() , 1, 1),
                new TensorOperationImpl<>("LossBackward", l-> { input.sub_(target); return input;  },
                        input.size()));

        return output;
    }
}
