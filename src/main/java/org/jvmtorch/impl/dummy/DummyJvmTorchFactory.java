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
package org.jvmtorch.impl.dummy;

import org.jvmtorch.JvmTorchFactory;
import org.jvmtorch.nn.NN;
import org.jvmtorch.nn.functional.Functional;
import org.jvmtorch.torch.Torch;
import org.jvmtorch.torch.optim.Optim;

public class DummyJvmTorchFactory implements JvmTorchFactory {

	@Override
	public Torch createTorch() {
		throw new UnsupportedOperationException("Not yet implemented");
	}

	@Override
	public Functional createFunctional() {
		throw new UnsupportedOperationException("Not yet implemented");
	}

	@Override
	public NN createNN() {
		throw new UnsupportedOperationException("Not yet implemented");
	}

	@Override
	public Optim createOptim() {
		throw new UnsupportedOperationException("Not yet implemented");
	}
}
