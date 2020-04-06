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
package org.jvmtorch.torch;

import org.jvmpy.python.GenericTuple;
import org.jvmpy.python.Tuple;

public abstract class NextFunctions<T extends TensorOperations<T>> extends GenericTuple<Tuple<GradFunction<T>>>{

	@SafeVarargs
	public NextFunctions(Tuple<GradFunction<T>> first, Tuple<GradFunction<T>>... remaining) {
		super(first, remaining);
	}
	
	@SuppressWarnings("unchecked")
	@Override
	public GenericTuple<GradFunction<T>>[] getComponents() {
		return getComponentsAsType(GenericTuple.class);
	}
}
