package org.jvmtorch.torch;

import static org.jvmpy.python.Python.len;
import static org.jvmtorch.JvmTorch.torch;

import org.junit.Assert;
import org.junit.Test;
import org.jvmtorch.testing.TestCase;

public class AutogradTest extends TestCase<AutogradTest> {
	
	@Test
	public void test_scalartensor_addition() {
		var a = torch.randn(2, 2).requires_grad_(true);
		var b = torch.randn().requires_grad_(true);
		var c = a.add(b);
		
		self.assertTrue(a.requires_grad());
		self.assertTrue(b.requires_grad());
		
		c.backward(torch.ones(2, 2).mul(2f));

		self.assertTrue(b.grad().size().dimensions().length == 0);
		self.assertTrue(b.grad().numel() == 1);
		self.assertEqual(b.grad().item(), 8f);
		
		self.assertArrayEqual(a.grad().getDataAsFloatArray(), torch.ones(2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);
	}
	
	@Test
	public void test_scalartensor_addition_second_without_requires_grad() {
		var a = torch.randn(2, 2).requires_grad_(true);
		var b = torch.randn();
		var c = a.add(b);
		
		self.assertTrue(a.requires_grad());
		self.assertFalse(b.requires_grad());
		
		c.backward(torch.ones(2, 2).mul(2f));
		
		self.assertTrue(a.requires_grad());
		self.assertFalse(b.requires_grad());

		self.assertIsNone(b.grad());
		
		self.assertArrayEqual(a.grad().getDataAsFloatArray(), torch.ones(2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);
	}
	
	@Test
	public void test_scalartensor_addition_first_without_requires_grad() {
		var a = torch.randn(2, 2);
		var b = torch.randn().requires_grad_(true);
		var c = a.add(b);
		
		self.assertTrue(c.requires_grad());
		self.assertFalse(a.requires_grad());
		self.assertTrue(b.requires_grad());
		
		c.backward(torch.ones(2, 2).mul(2f));

		self.assertTrue(b.grad().size().dimensions().length == 0);
		self.assertTrue(b.grad().numel() == 1);
		self.assertEqual(b.grad().item(), 8f);
		
		self.assertIsNone(a.grad());
	}
	
	
	@Test
	public void test_scalartensor_addition_reversed() {
		var a = torch.randn(2, 2).requires_grad_(true);
		var b = torch.randn().requires_grad_(true);		
		
		var c = b.add(a);
		
		self.assertTrue(a.requires_grad());
		self.assertTrue(b.requires_grad());
		
		c.backward(torch.ones(2, 2).mul(2f));

		self.assertTrue(b.grad().size().dimensions().length == 0);
		self.assertTrue(b.grad().numel() == 1);
		self.assertEqual(b.grad().item(), 8f);
		
		self.assertArrayEqual(a.grad().getDataAsFloatArray(), torch.ones(2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);
	}
	
	//
	
	@Test
	public void test_both_scalartensor_addition() {
		var a = torch.randn().requires_grad_(true);
		var b = torch.randn().requires_grad_(true);
		var c = a.add(b);
		
		self.assertTrue(a.requires_grad());
		self.assertTrue(b.requires_grad());
		
		c.backward(torch.ones().mul(2f));

		self.assertTrue(b.grad().size().dimensions().length == 0);
		self.assertTrue(b.grad().numel() == 1);
		self.assertEqual(b.grad().item(), 2f);
		
		self.assertArrayEqual(a.grad().getDataAsFloatArray(), torch.ones().mul(2f).getDataAsFloatArray(), 0.0001f);
	}
	
	@Test
	public void test_both_scalartensor_addition_second_without_requires_grad() {
		var a = torch.randn().requires_grad_(true);
		var b = torch.randn();
		var c = a.add(b);
		
		self.assertTrue(a.requires_grad());
		self.assertFalse(b.requires_grad());
		
		c.backward(torch.ones().mul(2f));
		
		self.assertTrue(a.requires_grad());
		self.assertFalse(b.requires_grad());

		self.assertIsNone(b.grad());
		
		self.assertArrayEqual(a.grad().getDataAsFloatArray(), torch.ones().mul(2f).getDataAsFloatArray(), 0.0001f);
	}
	
	@Test
	public void test_both_scalartensor_addition_first_without_requires_grad() {
		var a = torch.randn();
		var b = torch.randn().requires_grad_(true);
		var c = a.add(b);
		
		self.assertTrue(c.requires_grad());
		self.assertFalse(a.requires_grad());
		self.assertTrue(b.requires_grad());
		
		c.backward(torch.ones().mul(2f));

		self.assertTrue(b.grad().size().dimensions().length == 0);
		self.assertTrue(b.grad().numel() == 1);
		self.assertEqual(b.grad().item(), 2f);
		
		self.assertIsNone(a.grad());
	}
	
	
	@Test
	public void test_both_scalartensor_addition_reversed() {
		var a = torch.randn().requires_grad_(true);
		var b = torch.randn().requires_grad_(true);		
		
		var c = b.add(a);
		
		self.assertTrue(a.requires_grad());
		self.assertTrue(b.requires_grad());
		
		c.backward(torch.ones().mul(2f));

		self.assertTrue(b.grad().size().dimensions().length == 0);
		self.assertTrue(b.grad().numel() == 1);
		self.assertEqual(b.grad().item(), 2f);
		
		self.assertArrayEqual(a.grad().getDataAsFloatArray(), torch.ones().mul(2f).getDataAsFloatArray(), 0.0001f);
	}
	
	@Test
	public void test_scalarbroadcast_addition() {
		var a = torch.randn(2, 2).requires_grad_(true);
		var b = torch.randn(1, 1).requires_grad_(true);
		var c = a.add(b);
		
		self.assertTrue(a.requires_grad());
		self.assertTrue(b.requires_grad());
		
		c.backward(torch.ones(2, 2).mul(2f));

		self.assertTrue(b.grad().size().dimensions().length == 2);
		self.assertTrue(b.grad().numel() == 1);
		self.assertEqual(b.grad().item(), 8f);
		
		self.assertArrayEqual(a.grad().getDataAsFloatArray(), torch.ones(2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);
	}
	
	@Test
	public void test_scalarbroadcast_addition_second_without_requires_grad() {
		var a = torch.randn(2, 2).requires_grad_(true);
		var b = torch.randn(1, 1);
		var c = a.add(b);
		
		self.assertTrue(a.requires_grad());
		self.assertFalse(b.requires_grad());
		
		c.backward(torch.ones(2, 2).mul(2f));
		
		self.assertTrue(a.requires_grad());
		self.assertFalse(b.requires_grad());

		self.assertIsNone(b.grad());
		
		self.assertArrayEqual(a.grad().getDataAsFloatArray(), torch.ones(2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);
	}
	
	@Test
	public void test_scalarbroadcast_addition_first_without_requires_grad() {
		var a = torch.randn(2, 2);
		var b = torch.randn(1, 1).requires_grad_(true);
		var c = a.add(b);
		
		self.assertTrue(c.requires_grad());
		self.assertFalse(a.requires_grad());
		self.assertTrue(b.requires_grad());
		
		c.backward(torch.ones(2, 2).mul(2f));

		self.assertTrue(b.grad().size().dimensions().length == 2);
		self.assertTrue(b.grad().numel() == 1);
		self.assertEqual(b.grad().item(), 8f);
		
		self.assertIsNone(a.grad());
	}
	
	
	@Test
	public void test_scalarbroadcast_addition_reversed() {
		var a = torch.randn(2, 2).requires_grad_(true);
		var b = torch.randn(1, 1).requires_grad_(true);		
		
		var c = b.add(a);
		
		self.assertTrue(a.requires_grad());
		self.assertTrue(b.requires_grad());
		
		c.backward(torch.ones(2, 2).mul(2f));

		self.assertTrue(b.grad().size().dimensions().length == 2);
		self.assertTrue(b.grad().numel() == 1);
		self.assertEqual(b.grad().item(), 8f);
		
		self.assertArrayEqual(a.grad().getDataAsFloatArray(), torch.ones(2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);
	}
	
	
	@Test
	public void test_tensor_addition() {
		var a = torch.randn(2, 2).requires_grad_(true);
		var b = torch.randn(2, 2).requires_grad_(true);
		var c = a.add(b);
		
		self.assertTrue(a.requires_grad());
		self.assertTrue(b.requires_grad());
		
		c.backward(torch.ones(2, 2).mul(2f));

		
		self.assertArrayEqual(a.grad().getDataAsFloatArray(), torch.ones(2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);
		self.assertArrayEqual(b.grad().getDataAsFloatArray(), torch.ones(2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);
	}
	
	@Test
	public void test_tensor_addition_second_without_requires_grad() {
		var a = torch.randn(2, 2).requires_grad_(true);
		var b = torch.randn(2, 2);
		var c = a.add(b);
		
		self.assertTrue(a.requires_grad());
		self.assertFalse(b.requires_grad());
		
		c.backward(torch.ones(2, 2).mul(2f));
		
		self.assertTrue(a.requires_grad());
		self.assertFalse(b.requires_grad());

		self.assertIsNone(b.grad());
		
		self.assertArrayEqual(a.grad().getDataAsFloatArray(), torch.ones(2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);
	}
	
	@Test
	public void test_tensor_addition_first_without_requires_grad() {
		var a = torch.randn(2, 2);
		var b = torch.randn(2, 2).requires_grad_(true);
		var c = a.add(b);
		
		self.assertTrue(c.requires_grad());
		self.assertFalse(a.requires_grad());
		self.assertTrue(b.requires_grad());
		
		c.backward(torch.ones(2, 2).mul(2f));

		self.assertArrayEqual(b.grad().getDataAsFloatArray(), torch.ones(2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);
		self.assertArrayEqual(b.grad().getDataAsFloatArray(), torch.ones(2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);
		
		self.assertIsNone(a.grad());
	}
	
	@Test
	public void test_tensor_addition_reversed() {
		var a = torch.randn(2, 2).requires_grad_(true);
		var b = torch.randn(2, 2).requires_grad_(true);
		var c = b.add(a);
		
		self.assertTrue(a.requires_grad());
		self.assertTrue(b.requires_grad());
		
		c.backward(torch.ones(2, 2).mul(2f));

		
		self.assertArrayEqual(a.grad().getDataAsFloatArray(), torch.ones(2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);
		self.assertArrayEqual(b.grad().getDataAsFloatArray(), torch.ones(2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);
	}

	
	@Test
	public void test_scalar_addition() {
		var a = torch.randn(2, 2).requires_grad_(true);
		var b = (float)Math.random();
		var c = a.add(b);
		
		self.assertTrue(a.requires_grad());
		
		c.backward(torch.ones(2, 2).mul(2f));
		
		self.assertTrue(a.requires_grad());

		self.assertArrayEqual(a.grad().getDataAsFloatArray(), torch.ones(2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);

	}
	
	@Test(expected = IllegalStateException.class)
	public void test_scalar_addition_without_requires_grad() {
		var a = torch.randn(2, 2);
		var b = (float)Math.random();
		var c = a.add(b);
		
		self.assertFalse(a.requires_grad());
		
		c.backward(torch.ones(2, 2).mul(2f));

	}
	

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

	
	@Test
	public void test_hessian_vector() {
		
	
	        var x = torch.rand(2, 2).requires_grad_(true);
	        var y = torch.rand(2, 2).requires_grad_(true);

	        var z = x.mul(x).add(y.mul(x).add(y.mul(y)));
	        z.backward(torch.ones(2, 2), true); // create_graph=True
	        
	        //with torch.no_grad():
	        x.requires_grad_(false);
	        y.requires_grad_(false);
	        
		        var x_grad = x.mul(2).add(y);
		        var y_grad = x.add(y.mul(2));
		 
		        self.assertArrayEqual(x.grad().getDataAsFloatArray(), x_grad.getDataAsFloatArray(), 0.0001f);
		        self.assertArrayEqual(y.grad().getDataAsFloatArray(), y_grad.getDataAsFloatArray(), 0.0001f);
		     
	        x.requires_grad_(true);
	        y.requires_grad_(true);
	        	        	        
	        var grad_sum = x.grad().mul(2).add(y.grad());
	        
	        grad_sum.backward(torch.ones(2, 2));
	        var x_hv = torch.ones(2, 2).mul(5); // Should be ones not zeros with create graph
	        var y_hv = torch.ones(2, 2).mul(4); // Should be ones not zeros with create graph

	        self.assertArrayEqual(x.grad().getDataAsFloatArray(), x_grad.add(x_hv).getDataAsFloatArray(), 0.0001f);
	        self.assertArrayEqual(y.grad().getDataAsFloatArray(), y_grad.add(y_hv).getDataAsFloatArray(), 0.0001f);
	}
	
	@Test(expected=IllegalStateException.class)
	public void test_hessian_vector_without_create_graph() {
		
	        var x = torch.rand(2, 2).requires_grad_(true);
	        var y = torch.rand(2, 2).requires_grad_(true);

	        var z = x.mul(x).add(y.mul(x).add(y.mul(y)));

	        z.backward(torch.ones(2, 2)); // create_graph=False
	        
	        //with torch.no_grad():
	        x.requires_grad_(false);
	        y.requires_grad_(false);
	        
		        var x_grad = x.mul(2).add(y);
		        var y_grad = x.add(y.mul(2));
		 
		        self.assertArrayEqual(x.grad().getDataAsFloatArray(), x_grad.getDataAsFloatArray(), 0.0001f);
		        self.assertArrayEqual(y.grad().getDataAsFloatArray(), y_grad.getDataAsFloatArray(), 0.0001f);
		     
	        x.requires_grad_(true);
	        y.requires_grad_(true);
	        
	        var grad_sum = x.grad().mul(2).add(y.grad());
	        
	        grad_sum.backward(torch.ones(2, 2));
	}


	@Override
	protected AutogradTest self() {
		return this;
	}

}
