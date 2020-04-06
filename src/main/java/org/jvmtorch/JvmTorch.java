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

import org.jvmtorch.impl.dummy.DummyJvmTorch;

/**
 * Entry point for JvmTorch functionality from Java code.
 * 
 * Developers can statically import the JvmTorch attributes
 * (ie. import static org.jvmpy.jvmtorch.JvmTorch.*; )
 * 
 * torch, F, nn and optim are then available for use,
 * 
 * eg.  var x = torch.randn(2, 3);
 * 
 * For alternate implementations, developers can create their
 * own JvmTorch class in a separate package extending from 
 * a custom implementation class.
 * 
 * This extension strategy means that only the package name of the
 * JvmTorch import needs to be changed to switch implementations.
 * 
 * Custom base classes can delegate to an implementation of
 * JvmTorchFactory which bridges the gap between Python-style 
 * coding (JvmTorch) and a Java-style factory pattern
 * ( JvmTorchFactory )
 * 
 * @author Michael Lavelle
 */
public class JvmTorch extends DummyJvmTorch {
}
