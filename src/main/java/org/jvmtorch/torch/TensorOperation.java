package org.jvmtorch.torch;

import org.jvmpy.symbolictensors.Operation;

public interface TensorOperation<T> extends Operation<T> {

    int[] dimensions();
}
