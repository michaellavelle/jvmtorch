package org.jvmpy.symbolictensors;

import java.util.function.UnaryOperator;

public class OperationImpl<T> implements Operation<T> {

    private UnaryOperator<T> operation;
    private int[] dimensions;
    private String name;

    public OperationImpl(String name, UnaryOperator<T> operation, int[] dimensions) {
        this.dimensions = dimensions;
        this.operation = operation;
        this.name = name;
    }

    public int[] dimensions() {
        return dimensions;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public T apply(T t) {
        return operation.apply(t);
    }
}
