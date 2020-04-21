package org.jvmtorch.impl.ml4j;

import java.util.Arrays;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.ml4j.nn.neurons.format.features.Dimension;
import org.ml4j.nn.neurons.format.features.DimensionScope;

public class NeuronsActivationSizeTest {
	
	@Test
	public void testExampleOnly() {
		
		DimensionScope scope = DimensionScope.INPUT;
		
		List<String> dimensionNames = Arrays.asList("example");	
		NeuronsActivationFormat<?> format = NeuronsActivationSize.getNeuronsActivationFormat(dimensionNames, scope);
		
		System.out.println(format);
		
		Assert.assertEquals(1, format.getExampleDimensions().size());
		Assert.assertEquals(1, format.getDimensions().size());
		
		Assert.assertEquals(Dimension.EXAMPLE, format.getExampleDimensions().get(0));
		Assert.assertEquals(Dimension.EXAMPLE, format.getDimensions().get(0));

		// TODO
		Assert.assertEquals(NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET, format.getFeatureOrientation());
	}
	
	@Test(expected = IllegalStateException.class)
	public void testFeatureOnly() {
		
		DimensionScope scope = DimensionScope.INPUT;
		
		List<String> dimensionNames = Arrays.asList("feature");	
		NeuronsActivationFormat<?> format = NeuronsActivationSize.getNeuronsActivationFormat(dimensionNames, scope);
		
		System.out.println(format);
		
		Assert.assertEquals(1, format.getExampleDimensions().size());
		Assert.assertEquals(1, format.getDimensions().size());
		
		Assert.assertEquals(Dimension.EXAMPLE, format.getExampleDimensions().get(0));
		Assert.assertEquals(Dimension.EXAMPLE, format.getDimensions().get(0));

		// TODO
		Assert.assertEquals(NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET, format.getFeatureOrientation());
	}
	
	@Test
	public void testExampleFeature() {
		
		DimensionScope scope = DimensionScope.INPUT;
		
		List<String> dimensionNames = Arrays.asList("example", "feature");	
		NeuronsActivationFormat<?> format = NeuronsActivationSize.getNeuronsActivationFormat(dimensionNames, scope);
		
		System.out.println(format);
		
		Assert.assertEquals(1, format.getExampleDimensions().size());
		Assert.assertEquals(2, format.getDimensions().size());
		
		Assert.assertEquals(Dimension.EXAMPLE, format.getExampleDimensions().get(0));
		Assert.assertEquals(Dimension.EXAMPLE, format.getDimensions().get(0));
		Assert.assertEquals(Dimension.FEATURE, format.getDimensions().get(1));


		Assert.assertEquals(NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET, format.getFeatureOrientation());
	}
	
	@Test
	public void testExampleInputFeature() {
		
		DimensionScope scope = DimensionScope.INPUT;
		
		List<String> dimensionNames = Arrays.asList("example", "input_feature");	
		NeuronsActivationFormat<?> format = NeuronsActivationSize.getNeuronsActivationFormat(dimensionNames, scope);
		
		System.out.println(format);
		
		Assert.assertEquals(1, format.getExampleDimensions().size());
		Assert.assertEquals(2, format.getDimensions().size());
		
		Assert.assertEquals(Dimension.EXAMPLE, format.getExampleDimensions().get(0));
		Assert.assertEquals(Dimension.EXAMPLE, format.getDimensions().get(0));
		Assert.assertEquals(Dimension.INPUT_FEATURE, format.getDimensions().get(1));


		Assert.assertEquals(NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET, format.getFeatureOrientation());
	}
	
	@Test
	public void testExampleOutputFeature() {
		
		DimensionScope scope = DimensionScope.INPUT;
		
		List<String> dimensionNames = Arrays.asList("example", "output_feature");	
		NeuronsActivationFormat<?> format = NeuronsActivationSize.getNeuronsActivationFormat(dimensionNames, scope);
		
		System.out.println(format);
		
		Assert.assertEquals(1, format.getExampleDimensions().size());
		Assert.assertEquals(2, format.getDimensions().size());
		
		Assert.assertEquals(Dimension.EXAMPLE, format.getExampleDimensions().get(0));
		Assert.assertEquals(Dimension.EXAMPLE, format.getDimensions().get(0));
		Assert.assertEquals(Dimension.OUTPUT_FEATURE, format.getDimensions().get(1));


		Assert.assertEquals(NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET, format.getFeatureOrientation());
	}
	
	@Test
	public void testFeatureExample() {
		
		DimensionScope scope = DimensionScope.INPUT;
		
		List<String> dimensionNames = Arrays.asList("feature", "example");	
		NeuronsActivationFormat<?> format = NeuronsActivationSize.getNeuronsActivationFormat(dimensionNames, scope);
		
		System.out.println(format);
		
		Assert.assertEquals(1, format.getExampleDimensions().size());
		Assert.assertEquals(2, format.getDimensions().size());
		
		Assert.assertEquals(Dimension.EXAMPLE, format.getExampleDimensions().get(0));
		Assert.assertEquals(Dimension.FEATURE, format.getDimensions().get(0));
		Assert.assertEquals(Dimension.EXAMPLE, format.getDimensions().get(1));


		Assert.assertEquals(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, format.getFeatureOrientation());
	}
	
	@Test
	public void testInputFeatureExample() {
		
		DimensionScope scope = DimensionScope.INPUT;
		
		List<String> dimensionNames = Arrays.asList("input_feature", "example");	
		NeuronsActivationFormat<?> format = NeuronsActivationSize.getNeuronsActivationFormat(dimensionNames, scope);
		
		System.out.println(format);
		
		Assert.assertEquals(1, format.getExampleDimensions().size());
		Assert.assertEquals(2, format.getDimensions().size());
		
		Assert.assertEquals(Dimension.EXAMPLE, format.getExampleDimensions().get(0));
		Assert.assertEquals(Dimension.INPUT_FEATURE, format.getDimensions().get(0));
		Assert.assertEquals(Dimension.EXAMPLE, format.getDimensions().get(1));


		Assert.assertEquals(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, format.getFeatureOrientation());
	}
	
	@Test
	public void testOutputFeatureExample() {
		
		DimensionScope scope = DimensionScope.INPUT;
		
		List<String> dimensionNames = Arrays.asList("output_feature", "example");	
		NeuronsActivationFormat<?> format = NeuronsActivationSize.getNeuronsActivationFormat(dimensionNames, scope);
		
		System.out.println(format);
		
		Assert.assertEquals(1, format.getExampleDimensions().size());
		Assert.assertEquals(2, format.getDimensions().size());
		
		Assert.assertEquals(Dimension.EXAMPLE, format.getExampleDimensions().get(0));
		Assert.assertEquals(Dimension.OUTPUT_FEATURE, format.getDimensions().get(0));
		Assert.assertEquals(Dimension.EXAMPLE, format.getDimensions().get(1));


		Assert.assertEquals(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, format.getFeatureOrientation());
	}
	
	@Test
	public void testExampleInputDepthInputHeightInputWidth() {
		
		DimensionScope scope = DimensionScope.INPUT;
		
		List<String> dimensionNames = Arrays.asList("example", "input_depth", "input_height", "input_width");	
		NeuronsActivationFormat<?> format = NeuronsActivationSize.getNeuronsActivationFormat(dimensionNames, scope);
		
		System.out.println(format);
		
		Assert.assertEquals(1, format.getExampleDimensions().size());
		Assert.assertEquals(4, format.getDimensions().size());
		
		Assert.assertEquals(Dimension.EXAMPLE, format.getExampleDimensions().get(0));
		Assert.assertEquals(Dimension.EXAMPLE, format.getDimensions().get(0));
		Assert.assertEquals(Dimension.INPUT_DEPTH, format.getDimensions().get(1));
		Assert.assertEquals(Dimension.INPUT_HEIGHT, format.getDimensions().get(2));
		Assert.assertEquals(Dimension.INPUT_WIDTH, format.getDimensions().get(3));




		Assert.assertEquals(NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET, format.getFeatureOrientation());
	}
	
	@Test
	public void testExampleInputDepthInputHeightInputWidthComposite() {
		
		DimensionScope scope = DimensionScope.INPUT;
		
		List<String> dimensionNames = Arrays.asList("example", "[input_depth, input_height, input_width]");	
		NeuronsActivationFormat<?> format = NeuronsActivationSize.getNeuronsActivationFormat(dimensionNames, scope);
		
		System.out.println(format);
		
		Assert.assertEquals(1, format.getExampleDimensions().size());
		Assert.assertEquals(2, format.getDimensions().size());
		
		Assert.assertEquals(Dimension.EXAMPLE, format.getExampleDimensions().get(0));
		Assert.assertEquals(Dimension.EXAMPLE, format.getDimensions().get(0));
		Assert.assertEquals(new Dimension.CompositeDimension(Arrays.asList(Dimension.INPUT_DEPTH, Dimension.INPUT_HEIGHT, Dimension.INPUT_WIDTH), DimensionScope.INPUT), format.getDimensions().get(1));

		Assert.assertEquals(NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET, format.getFeatureOrientation());
	}
	
	@Test
	public void testInputDepthInputHeightInputWidthCompositeExample() {
		
		DimensionScope scope = DimensionScope.INPUT;
		
		List<String> dimensionNames = Arrays.asList("[input_depth, input_height, input_width]", "example");	
		NeuronsActivationFormat<?> format = NeuronsActivationSize.getNeuronsActivationFormat(dimensionNames, scope);
		
		System.out.println(format);
		
		Assert.assertEquals(1, format.getExampleDimensions().size());
		Assert.assertEquals(2, format.getDimensions().size());
		
		Assert.assertEquals(Dimension.EXAMPLE, format.getExampleDimensions().get(0));
		Assert.assertEquals(new Dimension.CompositeDimension(Arrays.asList(Dimension.INPUT_DEPTH, Dimension.INPUT_HEIGHT, Dimension.INPUT_WIDTH), DimensionScope.INPUT), format.getDimensions().get(0));
		Assert.assertEquals(Dimension.EXAMPLE, format.getDimensions().get(1));

		Assert.assertEquals(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, format.getFeatureOrientation());
	}
	
	
	@Test
	public void testMl4jImToColConvFormat() {
		
		DimensionScope scope = DimensionScope.INPUT;
		
		List<String> dimensionNames = Arrays.asList("Depth", "Filter Height", "Filter Width", "Filter Positions", "Example");	
		NeuronsActivationFormat<?> format = NeuronsActivationSize.getNeuronsActivationFormat(dimensionNames, scope);
		
		System.out.println(format);
		
		Assert.assertEquals(2, format.getExampleDimensions().size());
		Assert.assertEquals(5, format.getDimensions().size());
		
		Assert.assertEquals(Dimension.FILTER_POSITIONS, format.getExampleDimensions().get(0));
		Assert.assertEquals(Dimension.EXAMPLE, format.getExampleDimensions().get(1));

		Assert.assertEquals(Dimension.FILTER_POSITIONS, format.getDimensions().get(3));
		Assert.assertEquals(Dimension.EXAMPLE, format.getDimensions().get(4));

		Assert.assertEquals(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, format.getFeatureOrientation());
	}
	
	@Test
	public void testMl4jImToColConvCompositeFormatA() {
		
		DimensionScope scope = DimensionScope.INPUT;
		
		List<String> dimensionNames = Arrays.asList("[Depth, Filter Height, Filter Width]", "[Filter Positions, Example]");	
		NeuronsActivationFormat<?> format = NeuronsActivationSize.getNeuronsActivationFormat(dimensionNames, scope);
		
		System.out.println(format);
		
		Assert.assertEquals(1, format.getExampleDimensions().size());
		Assert.assertEquals(2, format.getDimensions().size());
		
		Assert.assertEquals(new Dimension.CompositeDimension(Arrays.asList(Dimension.DEPTH, Dimension.FILTER_HEIGHT, Dimension.FILTER_WIDTH),  DimensionScope.ANY), format.getDimensions().get(0));
		Assert.assertEquals(new Dimension.CompositeDimension(Arrays.asList(Dimension.FILTER_POSITIONS, Dimension.EXAMPLE),  DimensionScope.ANY), format.getDimensions().get(1));

		Assert.assertEquals(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, format.getFeatureOrientation());
	}
	
	@Test
	public void testMl4jImToColConvCompositeFormatB() {
		
		DimensionScope scope = DimensionScope.INPUT;
		
		List<String> dimensionNames = Arrays.asList("[Filter Positions, Example]", "[Depth, Filter Height, Filter Width]");	
		NeuronsActivationFormat<?> format = NeuronsActivationSize.getNeuronsActivationFormat(dimensionNames, scope);
		
		System.out.println(format);
		
		Assert.assertEquals(1, format.getExampleDimensions().size());
		Assert.assertEquals(2, format.getDimensions().size());
		
		Assert.assertEquals(new Dimension.CompositeDimension(Arrays.asList(Dimension.FILTER_POSITIONS, Dimension.EXAMPLE),  DimensionScope.ANY), format.getDimensions().get(0));
		Assert.assertEquals(new Dimension.CompositeDimension(Arrays.asList(Dimension.DEPTH, Dimension.FILTER_HEIGHT, Dimension.FILTER_WIDTH),  DimensionScope.ANY), format.getDimensions().get(1));

		Assert.assertEquals(NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET, format.getFeatureOrientation());
	}
	
	@Test(expected = IllegalStateException.class)
	public void testMl4jImToColConvCompositeFormatInvalid() {
		
		DimensionScope scope = DimensionScope.INPUT;
		
		List<String> dimensionNames = Arrays.asList("[Depth, Filter Height]", "[Filter Width, Filter Positions, Example]");	
		NeuronsActivationSize.getNeuronsActivationFormat(dimensionNames, scope);
	}


}
