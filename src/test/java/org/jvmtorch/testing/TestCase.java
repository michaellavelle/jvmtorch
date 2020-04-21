package org.jvmtorch.testing;

import org.junit.Assert;
import org.jvmpy.python.PythonClass;


public abstract class TestCase<T extends TestCase<T>> extends PythonClass<T>{

	protected void assertTrue(boolean condition) {
		Assert.assertTrue(condition);
	}
	
	protected void assertEqual(Object actual, Object expected) {
		Assert.assertEquals(expected, actual);
	}
	
	protected void assertIs(Object actual, Object expected) {
		Assert.assertSame(expected, actual);
	}
	
	protected void assertIsNotNone(Object object) {
		Assert.assertNotNull(object);
	}

}