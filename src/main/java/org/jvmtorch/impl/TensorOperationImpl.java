package org.jvmtorch.impl;

import org.jvmtorch.torch.Size;
import org.jvmtorch.torch.TensorOperation;

import java.util.function.UnaryOperator;

public class TensorOperationImpl<T> implements TensorOperation<T> {

    private UnaryOperator<T> operation;
    private Size size;
    private String name;

    public TensorOperationImpl(String name, UnaryOperator<T> operation, Size size) {
        this.size = size;
        this.name = name;
        this.operation = operation;
    }

    public TensorOperationImpl(String name, UnaryOperator<T> operation, int firstDim, int... remainingDims) {
        this.size = new Size(firstDim, remainingDims);
        this.operation = operation;
        this.name = name;
    }

    @Override
    public int[] dimensions() {
         return size.getDimensions();
    }

    @Override
    public String name() {
        return name;
    }

    ;

    @Override
    public T apply(T t) {
        return operation.apply(t);
    }
}
