package org.jvmtorch.impl.ml4j;

import org.jvmtorch.nn.Linear;
import org.jvmtorch.nn.NN;
import org.jvmtorch.torch.Tensor;

public class ML4JLinear extends Linear<ML4JLinear, ML4JTensorOperations> {

    public ML4JLinear(NN<ML4JTensorOperations> nn, int in, int out) {
        super(nn, in, out);
    }

    @Override
    public Tensor<ML4JTensorOperations> forward(Tensor<ML4JTensorOperations> input) {
        return F.linear(input, self.weight, self.bias);
    }

    @Override
    public ML4JLinear self() {
        return this;
    }
}
