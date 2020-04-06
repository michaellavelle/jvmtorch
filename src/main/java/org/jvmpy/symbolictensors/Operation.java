package org.jvmpy.symbolictensors;

import java.util.function.UnaryOperator;

public interface Operation<T> extends UnaryOperator<T> {

    int[] dimensions();
    String name();

}
