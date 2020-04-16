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

import static org.jvmpy.python.Python.tuple;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.jvmpy.python.IntTuple;
import org.jvmpy.python.Tuple;

import com.codepoetics.protonpack.StreamUtils;

public class Size extends IntTuple {

	private int numel;
	public Size[] sizeComponents;
	private String name;
	private final List<Size> alternates;

	public Size() {
		super(new int[0]);
		alternates = new ArrayList<>();
		sizeComponents = new Size[0];
		this.numel = 1;
	}

	public Size(Size... components) {
		super(getDimensions(components));
		this.alternates = new ArrayList<>();
		this.sizeComponents = components;
		if (components.length == 1) {
			this.name = components[0].name;
			this.numel = components[0].numel;
		}
		this.numel = Arrays.stream(components).mapToInt(s -> s.numel).reduce(1, (l, r) -> l * r);
	}
	
	public Size(List<Size> components) {
		super(getDimensions(components));
		this.alternates = new ArrayList<>();
		if (components != null && components.size() == 1) {
			this.name = components.get(0).name;
			this.numel = components.get(0).numel;
		}
		this.sizeComponents = components.toArray(new Size[components.size()]);
		this.numel = Arrays.stream(this.sizeComponents).mapToInt(s -> s.numel).reduce(1, (l, r) -> l * r);
	}

	public Size(int... components) {
		this(IntStream.of(components).mapToObj(i -> new Size(i)).toArray(i -> new Size[i]));
	}

	public Size(int numel) {
		super(numel);
		this.alternates = new ArrayList<>();
		this.numel = numel;
		this.sizeComponents = new Size[] { this };
	}


	private Tuple<String> extractNames(String name) {
		if (name != null) {
			String[] parts = name.split(", ");
			List<String> names = new ArrayList<>();
			for (String part : parts) {
				names.add(part.replaceAll(",", "").trim());
			}
			return tuple(names);

		} else {
			return tuple(Arrays.asList("None"));
		}
	}

	public Tuple<String> dimensionNames() {
		if (sizeComponents.length == 1) {
			if (name != null) {
				return tuple(Arrays.asList(name));
			} else {
				return tuple(Arrays.asList("None"));
			}
		}
		if (this.components == null || this.components.length == 0) {
			return tuple(new ArrayList<>());
		}
		List<String> ns = new ArrayList<>();
		for (int i = 0; i < dimensions().length; i++) {
			ns.add("None");
		}
		// return tuple(ns);
		return tuple(
				decompose().stream().flatMap(s -> s.dimensionNames().asList().stream()).collect(Collectors.toList()));
	}

	public Size names_(Tuple<String> names) {
		if (names != null) {
			return names_(names, true);
		}

		return this;
	}

	private Size names_(Tuple<String> names, boolean extract) {
		List<Size> decomposed = decompose();
		if (names.length() != 0 && decomposed.size() != names.length()) {
			if (extract) {
				return names_(tuple(names.asList().stream().flatMap(n -> extractNames(n).asList().stream())
						.collect(Collectors.toList())), false);
			} else {
				throw new IllegalArgumentException(decomposed.size() + ":" + names.length());
			}
		} else {
			StreamUtils.zipWithIndex(decomposed.stream())
					.forEach(i -> i.getValue().name = names.get((int) i.getIndex()));
		}
		return this;
	}

	public Size t() {
		Size matrixSize = asMatrixSize();
		Size t = new Size(matrixSize.getSecondComponent(), matrixSize.getFirstComponent());
		if (matrixSize.dimensionNames() != null) {
		}
		// System.out.println("SIZE T OUT:" + t);

		return t;

	}

	public Size asMatrixSize() {
		return asMatrixSize(true);
	}

	public Size asMatrixSize(boolean throwException) {
		if (sizeComponents.length == 2) {
			return this;
		} else if (components == null || components.length == 0) {
			return new Size(1, 1);
		} else if (components.length == 2) {
			return new Size(new Size(components[0]), new Size(components[1]));
		} else if (components.length == 1) {
			return new Size(new Size(1), new Size(components[0]));
		} else if (!this.alternates.isEmpty()) {
			List<Size> reversedAlternates = new ArrayList<>();
			reversedAlternates.addAll(this.alternates);
			Collections.reverse(reversedAlternates);
			for (Size alternateSize : reversedAlternates) {
				Size matrixSize = alternateSize.asMatrixSize(false);
				if (matrixSize != null) {
					return matrixSize;
				}
			}
		}
		if (throwException) {
			throw new IllegalStateException("Size cannot be converted into matrix size");
		} else {
			return null;
		}

	}

	public Size getFirstComponent() {
		if (sizeComponents.length < 1) {
			throw new IllegalStateException("Size does not have a first component");
		}
		return sizeComponents[0];
	}

	public Size getSecondComponent() {
		if (sizeComponents.length < 2) {
			throw new IllegalStateException("Size does not have a second component");
		}
		return sizeComponents[1];
	}

	
	public int numel() {
		return numel;
	}

	public int len() {
		return sizeComponents.length;
	}

	public List<Size> decompose() {
		if (sizeComponents.length == 1) {
			return Arrays.asList(this);
		} else {
			return Arrays.stream(sizeComponents).flatMap(c -> c.decompose().stream()).collect(Collectors.toList());
		}
	}

	private List<Size> getTwoSplits() {
		List<Size> decomposed = decompose();
		List<Size> twoSplits = new ArrayList<>();
		for (int i = 1; i < decomposed.size(); i++) {
			List<Size> first = decomposed.subList(0, i);
			List<Size> second = decomposed.subList(i, decomposed.size());
			Size twoSplit = new Size(new Size(first.toArray(new Size[first.size()])),
					new Size(second.toArray(new Size[second.size()])));
			twoSplits.add(twoSplit);
		}
		return twoSplits;
	}

	private List<Integer> getDimensions() {
		if (this.components == null || this.components.length == 0) {
			return new ArrayList<>();
		}
		return decompose().stream().map(s -> s.numel()).collect(Collectors.toList());
	}

	private static int[] getDimensions(Size[] components) {
		List<Size> decomposed = Arrays.stream(components).flatMap(c -> c.decompose().stream())
				.collect(Collectors.toList());
		int[] dims = new int[decomposed.size()];
		for (int i = 0; i < decomposed.size(); i++) {
			dims[i] = decomposed.get(i).numel();
		}
		return dims;
	}

	private static int[] getDimensions(List<Size> components) {
		List<Size> decomposed = components.stream().flatMap(c -> c.decompose().stream()).collect(Collectors.toList());
		int[] dims = new int[decomposed.size()];
		for (int i = 0; i < decomposed.size(); i++) {
			dims[i] = decomposed.get(i).numel();
		}
		return dims;
	}

	public int[] dimensions() {
		List<Integer> dims = getDimensions();
		int[] ret = new int[dims.size()];
		for (int i = 0; i < ret.length; i++) {
			ret[i] = dims.get(i);
		}
		return ret;
	}

	@Override
	public String toString() {
		return "torch.Size([" + dimensionsString() + "], names=" + dimensionNames() + ")";
	}

	private String dimensionsString() {
		String s = getDimensions().toString();
		return s.substring(1, s.length() - 1);
	}

	public List<Size> getAlternates() {
		return alternates;
	}

	public Optional<SizeComponentMatch> getMatches(Size second) {

		// Can we split the first size into two, so that the last part is within the
		// second size.
		List<Size> firstTwoSplits = this.getTwoSplits();
		List<SizeComponentMatch> matches = new ArrayList<>();
		for (Size firstTwoSplit : firstTwoSplits) {
			Size lastPartOfFirst = firstTwoSplit.sizeComponents[1];
			Optional<SizeComponentMatch> match = isContainedWithinSecond(lastPartOfFirst, second,
					firstTwoSplit.sizeComponents[0]);
			if (match.isPresent()) {
				matches.add(match.get());
			}
		}

		List<Size> secondTwoSplits = second.getTwoSplits();
		// Can we split the second size into two, so that the first part of the second
		// is the end of the first

		for (Size secondTwoSplit : secondTwoSplits) {
			Size firstPartOfSecond = secondTwoSplit.sizeComponents[0];
			Optional<SizeComponentMatch> match = isEndingOfFirst(firstPartOfSecond, this,
					secondTwoSplit.sizeComponents[1]);
			if (match.isPresent()) {
				matches.add(match.get());
			}
		}
		if (matches.isEmpty()) {
			return Optional.empty();
		} else {
			Integer maxLength = null;
			SizeComponentMatch bestMatch = null;
			for (SizeComponentMatch match : matches) {
				int sharedLength = match.shared.len();
				if (maxLength == null || sharedLength > maxLength) {
					bestMatch = match;
					maxLength = sharedLength;
				}
			}
			return Optional.of(bestMatch);
		}
	}

	private Optional<SizeComponentMatch> isContainedWithinSecond(Size searchFor, Size source,
			Size firstComponentPrefix) {
		if (source.dimensionsString().contains(searchFor.dimensionsString())) {
			int ind = source.dimensionsString().indexOf(searchFor.dimensionsString());
			// throw new RuntimeException("Found:" + searchFor.dimensionsString() + " in " +
			// source.dimensionsString() + " at index:" + ind);
			if (ind == 0) {
				int firstComponentsCount = searchFor.sizeComponents.length;
				List<Size> secondDecomposed = source.decompose();
				Size remaining = new Size(secondDecomposed.subList(firstComponentsCount, secondDecomposed.size()));

				return Optional.of(new SizeComponentMatch(firstComponentPrefix, searchFor, remaining));
			} else {
				int firstComponentsCount = searchFor.sizeComponents.length;
				List<Size> prefixComponents = new ArrayList<>();
				List<Size> secondDecomposed = source.decompose();
				int ind2 = 0;
				while (prefixComponents.size() * 3 < ind) {
					Size s = secondDecomposed.get(ind2++);
					prefixComponents.add(s);
				}
				int from = firstComponentsCount + prefixComponents.size();
				if (from < secondDecomposed.size()) {
					Size remaining = new Size(secondDecomposed.subList(from, secondDecomposed.size()));

					Size secondComponentPrefix = new Size(prefixComponents);
					return Optional.of(
							new SizeComponentMatch(firstComponentPrefix, searchFor, remaining, secondComponentPrefix));

				} else {
					return Optional.empty();
				}

			}
		} else {
			return Optional.empty();
		}
	}

	private Optional<SizeComponentMatch> isEndingOfFirst(Size searchFor, Size source, Size secondComponentSuffix) {
		if (source.dimensionsString().endsWith(searchFor.dimensionsString())) {
			int ind = source.dimensionsString().lastIndexOf(searchFor.dimensionsString());
			int ind2 = 0;
			List<Size> prefixComponents = new ArrayList<>();
			List<Size> firstDecomposed = source.decompose();

			while (prefixComponents.size() * 3 < ind) {
				Size s = firstDecomposed.get(ind2++);
				prefixComponents.add(s);
			}

			return Optional.of(new SizeComponentMatch(new Size(prefixComponents), searchFor, secondComponentSuffix));

		} else {
			return isProductOfEndingOfFirst(searchFor, source, secondComponentSuffix);
		}
	}

	private Optional<SizeComponentMatch> isProductOfEndingOfFirst(Size searchFor, Size source,
			Size secondComponentSuffix) {
		int[] firstDimensions = source.dimensions();
		for (int i = 0; i < firstDimensions.length; i++) {

			int firstEndingProduct = 1;
			int[] firstEnding = new int[i + 1];
			for (int j = 0; j < firstEnding.length; j++) {
				firstEnding[j] = firstDimensions[firstDimensions.length - 1 - i + j];
				firstEndingProduct = firstEndingProduct * firstEnding[j];
			}

			Size firstEndingSize = new Size(firstEnding);
			Size firstEndingProductSize = new Size(firstEndingProduct).names_(tuple("feature"));

			if (firstEndingProductSize.dimensionsString().equals(searchFor.dimensionsString())) {
				int ind = source.dimensionsString().lastIndexOf(firstEndingSize.dimensionsString());
				int ind2 = 0;
				List<Size> prefixComponents = new ArrayList<>();
				List<Size> firstDecomposed = source.decompose();

				while (prefixComponents.size() * 3 < ind) {
					Size s = firstDecomposed.get(ind2++);
					prefixComponents.add(s);
				}
				Size alternate = new Size(new Size(prefixComponents), firstEndingProductSize);
				alternates.add(alternate);

				SizeComponentMatch match = new SizeComponentMatch(new Size(prefixComponents), firstEndingProductSize,
						secondComponentSuffix);

				return Optional.of(match);

			}
		}
		return Optional.empty();

	}

	class SizeComponentMatch {

		Size firstComponentPrefix;
		Size shared;
		Optional<Size> secondComponentPrefix;
		Size secondComponentSuffix;

		public SizeComponentMatch(Size firstComponentPrefix, Size shared, Size secondComponentSuffix) {
			this.firstComponentPrefix = firstComponentPrefix;
			this.secondComponentPrefix = Optional.empty();
			this.secondComponentSuffix = secondComponentSuffix;
			this.shared = shared;
		}

		public SizeComponentMatch(Size firstComponentPrefix, Size shared, Size secondComponentSuffix,
				Size secondComponentPrefix) {
			this.firstComponentPrefix = firstComponentPrefix;
			this.secondComponentPrefix = Optional.of(secondComponentPrefix);
			this.secondComponentSuffix = secondComponentSuffix;
			this.shared = shared;
		}

		@Override
		public String toString() {
			return "SizeComponentMatch [firstComponentPrefix=" + firstComponentPrefix + ", shared=" + shared
					+ ", secondComponentPrefix=" + secondComponentPrefix + ", secondComponentSuffix="
					+ secondComponentSuffix + "]";
		}

	}

	public Size matmul(Size other) {
		Optional<SizeComponentMatch> match = getMatches(other);
		if (match.isPresent()) {
			SizeComponentMatch m = match.get();
			Size first = m.firstComponentPrefix;
			Size second = m.secondComponentSuffix;
			if (m.secondComponentPrefix.isPresent()) {
				second = new Size(m.secondComponentPrefix.get(), second);
			}
			Size ret = new Size(first, second);
			return ret;
		}

		throw new IllegalArgumentException("Unable to match sizes");
	}

}
