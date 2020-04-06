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

import org.jvmtorch.torch.Tensor;

public interface ModuleAttributes {

	default int num_flat_features(Tensor<?> x) {
		// int size = x.size()[1:] # all dimensions except the batch dimension
		int[] size = x.size().getDimensions();
		var num_features = 1;
		for (var s : size)
			num_features *= s;
		return num_features;
	}
}
