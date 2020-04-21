package org.jvmtorch.torch;

import org.junit.Assert;
import org.junit.Test;


public class SizeTest {

	@Test
	public void testConstructorForScalar() {
		Size size = new Size();
		Assert.assertNotNull(size.decompose());
		Assert.assertTrue(size.decompose().isEmpty());

		Assert.assertNotNull(size.asList());
		Assert.assertTrue(size.asList().isEmpty());
		
		Assert.assertNotNull(size.dimensions());
		Assert.assertTrue(size.dimensions().length == 0);

		Assert.assertNotNull(size.dimensionNames());
		Assert.assertTrue(size.dimensionNames().length() == 0);

		Assert.assertNotNull(size.dimensionNames().asList());
		Assert.assertTrue(size.dimensionNames().asList().isEmpty());

		Assert.assertNotNull(size.sizeComponents);
		Assert.assertTrue(size.sizeComponents.length == 0);

		Assert.assertNotNull(size.asMatrixSize());
		Assert.assertNotNull(size.asMatrixSize().equals(size));

		Assert.assertNotNull(size.getAlternates());
		Assert.assertTrue(size.getAlternates().isEmpty());

		Assert.assertNotNull(size.getComponents());
		Assert.assertTrue(size.getComponents().length == 0);

		Assert.assertEquals(0, size.len());
		
		Assert.assertEquals(0, size.length());
		
		Assert.assertEquals(1, size.numel());

	}
	
	@Test
	public void testConstructorForVector() {
		Size size = new Size(10);
		Assert.assertNotNull(size.decompose());
		Assert.assertTrue(!size.decompose().isEmpty());
		Assert.assertTrue(size.decompose().size() == 1);
		Assert.assertEquals(size, size.decompose().get(0));

		
		Assert.assertNotNull(size.asList());
		Assert.assertTrue(!size.asList().isEmpty());
		Assert.assertTrue(size.asList().size() == 1);

		
		Assert.assertNotNull(size.dimensions());
		Assert.assertTrue(size.dimensions().length == 1);
		Assert.assertEquals(10, size.dimensions()[0]);

		Assert.assertNotNull(size.dimensionNames());
		Assert.assertTrue(size.dimensionNames().length() == 1);
		Assert.assertEquals("None", size.dimensionNames().get(0));

		
		Assert.assertNotNull(size.dimensionNames().asList());
		Assert.assertTrue(!size.dimensionNames().asList().isEmpty());
		Assert.assertEquals("None", size.dimensionNames().asList().get(0));


		Assert.assertNotNull(size.sizeComponents);
		Assert.assertTrue(size.sizeComponents.length == 1);
		Assert.assertEquals(size, size.sizeComponents[0]);


		Assert.assertNotNull(size.asMatrixSize());
		//Assert.assertEquals(new Size(1, 10), size.asMatrixSize());

		Assert.assertNotNull(size.getAlternates());
		Assert.assertTrue(size.getAlternates().isEmpty());

		Assert.assertNotNull(size.getComponents());
		Assert.assertTrue(size.getComponents().length == 1);

		Assert.assertEquals(1, size.len());
		
		Assert.assertEquals(1, size.length());
		
		Assert.assertEquals(10, size.numel());

	}
}
