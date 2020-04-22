package org.jvmtorch.torch;

import static org.jvmpy.python.Python.list;
import static org.jvmtorch.JvmTorch.nn;

import org.junit.Test;
import org.jvmtorch.nn.IModule;
import org.jvmtorch.nn.Linear;
import org.jvmtorch.nn.Module;
import org.jvmtorch.nn.NN;
import org.jvmtorch.testing.TestCase;


public class NNTest extends TestCase<NNTest> {

	@Override
	protected NNTest self() {
		return this;
	}
	
	@Test
    public void test_modules() {
    	
        var l = nn.Linear(10, 20);
    	
    	@SuppressWarnings("unused")
        class Net extends Module<Net> {
        
        	private Linear<?> l1;
        	private Linear<?> l2;
			private Tensor param;
 
        	public Net(NN nn) {
				super(nn);
				 self.l1 = l;
			     self.l2 = l;
			     self.param = torch.empty(3, 5);
			}

			@Override
			public Tensor forward(Tensor input) {
				throw new UnsupportedOperationException("no-op");
			}
			
			@Override
			protected Net self() {
				return this;
			}
        }

        var n = new Net(nn);
        var s = nn.Sequential(n, n, n, n);
        self.assertArrayEqual(list(s.modules(), IModule.class), new IModule[] {s, n, l});
    }
	
}
