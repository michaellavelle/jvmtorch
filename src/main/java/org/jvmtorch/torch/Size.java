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

import org.jvmpy.python.IntTuple;

import java.util.Arrays;

public class Size extends IntTuple {

	public Size(int first, int...remaining) {
		super(first, remaining);
		if (first == 0) {
			throw new RuntimeException("First is 0");
		}
		for (int r : remaining) {
			if (r == 0) {
				throw new RuntimeException("One of remaining is 0");
			}
		}
	}

	@Override
	public String toString() {
		return "torch.Size(" + Arrays.asList(this.getComponents()) + ")";
	}

	@Override
	public Integer get(int first, int... remaining) {
		// TODO
		return (Integer) super.get(first, remaining);
	}

	public int[] getDimensions() {
		int[] dimensions = new int[this.getComponents().length];
		int index = 0;
		for (Integer dimension : this.getComponents()) {
			dimensions[index++] = dimension;
			if (dimension == 0) {
				throw new RuntimeException();
			}
		}
		return dimensions;
	}
	
}
