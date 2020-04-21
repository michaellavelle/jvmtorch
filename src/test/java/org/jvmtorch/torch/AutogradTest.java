package org.jvmtorch.torch;

import static org.jvmpy.python.Python.len;
import static org.jvmpy.python.Python.print;
import static org.jvmtorch.JvmTorch.torch;

import org.junit.Assert;
import org.junit.Test;
import org.jvmtorch.testing.TestCase;

public class AutogradTest extends TestCase<AutogradTest> {

	@Test
	public void test_requires_grad_inplace() {
		var a = torch.randn(5, 5);
		var b = torch.randn(5, 5).requires_grad_(true);
		a = a.add(b);
		
		self.assertTrue(a.requires_grad());

		// non-leaf
		a = torch.randn(5, 5).add(0f);
		b = torch.randn(5, 5).requires_grad_(true);
		a = a.add(b);
		self.assertTrue(a.requires_grad());
	}

	/**
	 * TODO - fix this test.
	 */
	@Test
	public void test_next_functions() {
		var x = torch.randn(5, 5).requires_grad_(true);
		var y = torch.randn(5, 5).requires_grad_(true);

		var a = x.add(y);
		self.assertIsNotNone(a.grad_fn());
		var next_functions = a.grad_fn().next_functions();
		Assert.assertEquals(len(next_functions), 2);
		print(next_functions.get(0, 0));
		self.assertEqual(next_functions.get(0, 0).toString(), "<AccumulateGrad object>");
		//self.assertEqual(next_functions.get(0,  1), 0);  // TODO - uncomment
		self.assertEqual(next_functions.get(1, 0).toString(), "<AccumulateGrad object>");
		//self.assertEqual(next_functions.get(1,  1), 0); // TODO - uncomment
	
		var b = a.add(5);
		next_functions = b.grad_fn().next_functions();
		self.assertEqual(len(next_functions), 1); // TODO // Should be 2
		self.assertIs(next_functions.get(0,  0), a.grad_fn());
		//Assert.assertNull(next_functions.get(1, 0)); // TODO - uncomment

	}

	@Override
	protected AutogradTest self() {
		return this;
	}

}
