package org.jvmtorch.torch;

import org.junit.Assert;
import org.junit.Test;
import static org.jvmpy.python.Python.*;

public class SizeMatcherTest {
	
	@Test
	public void testA() {
		
		Size first = new Size(3, 4).names_(tuple("3","4"));
		Size second = new Size(4, 5).names_(tuple("4","5"));
		
		Size result = SizeMatcher.matmul(first, second);
		
		Assert.assertNotNull(result);
		Assert.assertEquals(2, result.dimensions().length);
		Assert.assertEquals((int)3, (int)result.get(0));
		Assert.assertEquals((int)5, (int)result.get(1));
		
		System.out.println(result);

		
	}
	
	@Test
	public void testB() {
		
		Size first = new Size(3, 4, 5).names_(tuple("3","4", "5"));
		Size second = new Size(4, 5).names_(tuple("4","5"));
		
		Size result = SizeMatcher.matmul(first, second);
		
		Assert.assertEquals(1, result.dimensions().length);

		Assert.assertNotNull(result);
		Assert.assertEquals((int)3, (int)result.get(0));
		
		System.out.println(result);

		
	}
	
	@Test // ??
	public void testC() {
		
		Size first = new Size(3, 1, 4, 5).names_(tuple("3", "1", "4","5"));
		Size second = new Size(4, 5, 6).names_(tuple("4","5", "6"));
		
		Size result = SizeMatcher.matmul(first, second);
		
		Assert.assertEquals(3, result.dimensions().length);
		
		Assert.assertNotNull(result);
		Assert.assertEquals((int)3, (int)result.get(0));
		Assert.assertEquals((int)1, (int)result.get(1));
		Assert.assertEquals((int)6, (int)result.get(2));
		
		System.out.println(result);

		
	}
	
	@Test
	public void testC2() {
		
		Size first = new Size(3, 1, 4, 5).names_(tuple("3", "1", "4","5"));
		Size second = new Size(1, 4, 5, 6).names_(tuple("1", "4","5", "6"));
		
		Size result = SizeMatcher.matmul(first, second);
		
		Assert.assertEquals(2, result.dimensions().length);
		
		Assert.assertNotNull(result);
		Assert.assertEquals((int)3, (int)result.get(0));
		Assert.assertEquals((int)6, (int)result.get(1));
		
		System.out.println(result);

		
	}
	
	@Test
	public void testD() {
		
		Size first = new Size(3, 4, 5).names_(tuple("3","4", "5"));;
		Size second = new Size(4, 5, 6).names_(tuple("4","5", "6"));;
		
		Size result = SizeMatcher.matmul(first, second);
		
		Assert.assertEquals(2, result.dimensions().length);
		
		Assert.assertNotNull(result);
		Assert.assertEquals((int)3, (int)result.get(0));
		Assert.assertEquals((int)6, (int)result.get(1));
		
		System.out.println(result);

		
	}
	
	@Test // ?? - Should this be 3, 1, 6 ?
	public void testE() {
		
		Size first = new Size(3, 1, 4, 5).names_(tuple("3","1", "4", "5"));
		Size second = new Size(20, 6).names_(tuple("20","6"));
		
		Size result = SizeMatcher.matmul(first, second);
		
		Assert.assertEquals(2, result.dimensions().length);
		
		Assert.assertNotNull(result);
		Assert.assertEquals((int)3, (int)result.get(0));
		Assert.assertEquals((int)6, (int)result.get(1));
		
		System.out.println(result);
		
		System.out.println(first.getAlternates());

	}
	
	@Test
	public void testF() {
		
		Size first = new Size(3, 4, 5).names_(tuple("3","4", "5"));
		Size second = new Size(20, 6, 7).names_(tuple("20","6", "7"));
		
		Size result = SizeMatcher.matmul(first, second);
		
		Assert.assertEquals(3, result.dimensions().length);
		
		Assert.assertNotNull(result);
		Assert.assertEquals((int)3, (int)result.get(0));
		Assert.assertEquals((int)6, (int)result.get(1));
		Assert.assertEquals((int)7, (int)result.get(2));
		
		System.out.println(result);
		
		System.out.println(first.getAlternates());
		
	}
	
	

}
