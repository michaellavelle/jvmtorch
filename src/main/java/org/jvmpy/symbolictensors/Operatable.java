package org.jvmpy.symbolictensors;

public interface Operatable<T, O> {

    /**
     * Perform an inline operation on the underlying tensor,
     * potentially lazily.
     *
     * @param operation The operation to perform.
     */
    void performInlineOperation(Operation<T> operation);

    /**
     * Perform an operation
     *
     * @param newTensorName
     * @param operation
     * @return
     */
    O performUnaryMappingOperation(String newTensorName, Operation<T> operation);
}
