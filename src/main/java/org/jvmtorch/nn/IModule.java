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
package org.jvmtorch.nn;

import java.util.function.Function;

import org.jvmpy.python.OrderedDict;
import org.jvmtorch.torch.Tensor;

public interface IModule extends BaseModule, Function<Tensor, Tensor>{
	
	Tensor forward(Tensor input);
		
	OrderedDict<Parameter> parameters();
	
	void zero_grad();
}
