package org.jvmtorch.torch;

import org.jvmpy.symbolictensors.Operatable;

public interface ProviderTensorOperations<T, O> extends TensorOperations<T>, Operatable<T, O> {
    int[] getDimensions();
}
