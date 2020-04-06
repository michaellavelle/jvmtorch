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
package org.jvmpy.python;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class Python {
	
	public static boolean True = true;
	
	public static <T> void print(T o) {
		System.out.println(o == null ? "None" : o.toString());
	}
	
	public static <T> void print(T[] o) {
		System.out.println(o == null ? "None" : Arrays.asList(o).toString());
	}

	public static Tuple<Integer> inttuple(int first, int...remaining) {
		return new IntTuple(first, remaining);
	}

	@SafeVarargs
	public static <T> Tuple<T> tuple(T first, T...components) {
		return new GenericTuple<T>(first, components);
	}

	@SuppressWarnings("unchecked")
	public static <T> T[] list(OrderedDict<T> orderedDict) {
		List<T> list = new ArrayList<>();
		orderedDict.forEach(i -> list.add(i.getRight()));
		return list.toArray((T[])Array.newInstance(orderedDict.getElementClass(), list.size()));
	}
	
	public static <T> int len(T[] a) {
		return a.length;
	}
	
	public static <T> int len(Tuple<T> a) {
		return a.getComponents().length;
	}
	
	public static <T> String type(T type) {
		return "<class '" + type.getClass().getSimpleName() + "'>";
	}
	
	public static String type(Tuple<?> type) {
		return "<class 'tuple'>";
	}
}
