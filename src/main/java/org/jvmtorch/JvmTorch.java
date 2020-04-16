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
package org.jvmtorch;

import org.jvmtorch.impl.ml4j.ML4JJvmTorchFactory;
import org.jvmtorch.nn.NN;
import org.jvmtorch.nn.functional.Functional;
import org.jvmtorch.torch.Size;
import org.jvmtorch.torch.Torch;
import org.jvmtorch.torch.optim.Optim;

public class JvmTorch {
    
    public static final JvmTorchFactory DEFAULT_PYTORCH_FACTORY = new ML4JJvmTorchFactory();

    public static Torch torch;
    public static Functional F;
    public static NN nn;
    public static Optim optim;

    static {
    	init(DEFAULT_PYTORCH_FACTORY);
    }
    
    public static void init(JvmTorchFactory jvmTorchFactory) {
    	 torch = jvmTorchFactory.createTorch();
         F = jvmTorchFactory.createFunctional();
         nn = jvmTorchFactory.createNN();
         optim = jvmTorchFactory.createOptim();
    }
    
    public static Size Size(int... sizes) {
		return torch.Size(sizes);
	}
    
    public static Size Size(Size... sizes) {
		return torch.Size(sizes);
	}
}
