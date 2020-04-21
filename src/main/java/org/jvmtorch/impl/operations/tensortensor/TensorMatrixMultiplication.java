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

import java.util.List;
import java.util.function.UnaryOperator;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.jvmtorch.torch.Size;
import org.jvmtorch.torch.TensorOperations;
import org.jvmtorch.torch.Torch;

public class TensorMatrixMultiplication<T extends TensorOperations<T>> extends DifferentiableTensorTensorFunctionBase<T> {

	public TensorMatrixMultiplication(Torch torch) {
		super(torch);
	}
	
	/**
	 * Returns a UnaryOperator representing the function that matrix-multiplies the its input tensor
	 * by the provided tensor multiplier.  
	 * 
	 * @param <S>
	 * @param rightVariable
	 * @return
	 */
	public <S extends TensorOperations<S>> UnaryOperator<S> forwardPropFunction(S tensorMultiplier) {
		return t -> t.matmul(tensorMultiplier);
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
		return new ImmutablePair<>(g -> back(g, variables), 
				g -> g.t().matmul(convertToAlternativeSize(variables.getLeft()))
				.t());
	}
	
	
	
	private <S extends TensorOperations<S>>  S back(S g, Pair<S, S> variables) {
		
		// Back:torch.Size([1000, 10], names=(example, output_feature))

		// V1:torch.Size([1000, 100], names=(example, feature))
		// V2:torch.Size([100, 10], names=(input_feature, output_feature))

		//TH:10:1000
		//OTH:10:100
		
		// When evaluated:
		
		// g = 10:1000,,   should be 1000:10
		// variables.getRight().t() = 10:100  
		
		
		S s =  g.matmul(variables.getRight().t());
		return s;
	}

	@Override
	public UnaryOperator<Size> sizeFunction(Pair<Size, Size> variableSizes) {
		return s -> variableSizes.getLeft().matmul(variableSizes.getRight());
	}

	@Override
	public String name() {
		return "Mmul";
	}

	protected <S extends TensorOperations<S>> S convertToAlternativeSize(S tensor) {
		
		
		List<Size> alternatives = tensor.size().getAlternates();
		if (!alternatives.isEmpty()) {
			tensor = tensor.view(alternatives.get(alternatives.size() - 1));
		}


		return tensor;
	}

}
