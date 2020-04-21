package org.jvmtorch.impl.ml4j;

import java.util.ArrayList;
import java.util.List;

import org.jvmtorch.impl.TensorOperationImpl;
import org.jvmtorch.nn.Linear;
import org.jvmtorch.nn.NN;
import org.jvmtorch.torch.Size;
import org.jvmtorch.torch.Tensor;

public class ML4JLinear extends Linear<ML4JLinear> {
	
    public ML4JLinear(NN nn, int in, int out) {
        super(nn, in, out);
    }
  
    @Override
    public Tensor forward(Tensor input) {
    	
    	boolean needToTranspose = needToTranspose(input.size());
    	
    	Tensor in = input;
    	if (needToTranspose) {
    		in = input.t();
    	}
    	
    	if (input.size().dimensionNames() == null || input.size().dimensionNames().length() == 0) {
    		throw new IllegalArgumentException("Linear forward incorrect format");
    	}
    	
    	if (!toScopeIndependentNamesList(in.size().dimensionNames().asList()).get(0).equals("example")) {
    		throw new IllegalArgumentException("Linear forward incorrect format:" + input.size().dimensionNames());
    	}
    	
    	Tensor ret =  F.linear(in, self.weight, self.bias);

        Tensor output = ret.performUnaryMappingOperation(new TensorOperationImpl<>(torch, "LinearOutput", l -> l, s -> s), new TensorOperationImpl<>(torch, "LinearBackward", l-> backward(l), s ->  needToTranspose(s) ? s.t() : s));

    	return output;
    }
    
    private boolean needToTransposeBackward(Tensor l) {
    	boolean needToTranspose = false;
    	List<String> names = l.names().asList();
    	if (!names.get(0).equals("example")) {
    		needToTranspose = true;
    	}
    	return needToTranspose;
    }
    
    private boolean needToTranspose(Size size) {
    	boolean needToTranspose = false;
    	List<String> names = size.dimensionNames().asList();
    	if (!names.get(0).equals("example")) {
    		needToTranspose = true;
    	}

    	return needToTranspose;
    }

    private Tensor backward(Tensor tensor) {    
    	return needToTransposeBackward(tensor) ? tensor.t() : tensor;
	}
    
    
    private List<String> toScopeIndependentNamesList(List<String> strings) {
		List<String> returnValues = new ArrayList<>();
		for (String s : strings) {
			s = s.replaceAll("input_", "");
			s = s.replaceAll("output_", "");
			returnValues.add(s);
		}
		return returnValues;
	}


	@Override
    public ML4JLinear self() {
        return this;
    }
}
