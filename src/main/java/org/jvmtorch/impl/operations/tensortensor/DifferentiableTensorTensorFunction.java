/*
 * Copyright 2020 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */
package org.jvmtorch.impl.operations.tensortensor;

import java.util.function.UnaryOperator;

import org.apache.commons.lang3.tuple.Pair;
import org.jvmtorch.torch.Size;
import org.jvmtorch.torch.TensorOperations;

import com.google.common.base.Function;

public interface DifferentiableTensorTensorFunction<T extends TensorOperations<T>> extends Function<Pair<T, T>, T> {
	
	/**
	 * Uses the forwardPropFunction to add the left hand tensor of the specified pair of tensors
	 * to the right hand tensor.
	 */
	@Override
	default T apply(Pair<T, T> variables) {
		return forwardPropFunction(variables.getLeft()).apply(variables.getRight());
	}
	
	/**
	 * Returns a UnaryOperator representing the function that adds provided tensor delta to its input tensor
	 * 
	 * @param <S> The type of TensorOperations
	 * @param tensorDelta The tensor delta to add to the returned function's input tensor. 
	 * @return A function that adds provided tensor delta to its input tensor
	 */
	<S extends TensorOperations<S>> UnaryOperator<S> forwardPropFunction(S tensorDelta);

	
	/**
	 * Returns a pair of back propagation functions wrapping the partial derivatives wrt the left hand and
	 * right hand tensor - used to back propagate the gradient of an outer composite function to the gradient of 
	 * the composite function wrt each tensor.
	 * 
	 * @param <S>
	 * @param variables
	 * @return
	 */
	<S extends TensorOperations<S>> Pair<UnaryOperator<S>, UnaryOperator<S>> backPropFunctions(Pair<S, S> variables);
	
	
	/**
	 * Returns a function that calculates the size resulting from the addition of a tensor with size deltaSize
	 * to a tensor with the same size as it's input size. 
	 * 
	 * @param <S>
	 * @param variables
	 * @return
	 */
	UnaryOperator<Size> sizeFunction(Pair<Size, Size> variableSizes);
	
	
	String name();


}
