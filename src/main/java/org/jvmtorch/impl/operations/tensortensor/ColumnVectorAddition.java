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

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.jvmtorch.torch.Size;
import org.jvmtorch.torch.TensorOperations;
import org.jvmtorch.torch.Torch;

public class ColumnVectorAddition<T extends TensorOperations<T>> extends DifferentiableTensorTensorFunctionBase<T> {
	
	public ColumnVectorAddition(Torch torch) {
		super(torch);
	}
	
	/**
	 * Returns a UnaryOperator representing the function that add to the input tensor
	 * the provided tensor delta.  
	 * 
	 * @param <S>
	 * @param rightVariable
	 * @return
	 */
	public <S extends TensorOperations<S>> UnaryOperator<S> forwardPropFunction(S tensorDelta) {
			return t -> t.add(tensorDelta);
	}
	
	/**
	 * Returns a pair of back propagation functions wrapping the partial derivatives wrt the left hand and
	 * right hand tensor in order to back propagate the gradient of an outer composite function to the gradient of 
	 * the composite function wrt each tensor.
	 * 
	 * @param <S>
	 * @param variables
	 * @return
	 */
	public <S extends TensorOperations<S>> Pair<UnaryOperator<S>, UnaryOperator<S>> backPropFunctions(Pair<S, S> variables) {
		return new ImmutablePair<>(g -> g, g -> g.rowSums());
	}

	@Override
	public UnaryOperator<Size> sizeFunction(Pair<Size, Size> variableSizes) {
		return s -> variableSizes.getLeft();
	}

	@Override
	public String name() {
		return "AddColumnVector";
	}

}
