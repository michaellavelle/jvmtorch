package org.jvmtorch.impl.ml4j;

import org.jvmtorch.impl.TensorBase;
import org.jvmtorch.impl.TensorOperationImpl;
import org.jvmtorch.torch.Size;
import org.jvmtorch.torch.Tensor;
import org.jvmtorch.torch.Torch;
import org.jvmpy.symbolictensors.SymbolicTensor;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;

import java.util.ArrayList;
import java.util.List;

public class ML4JTensor extends TensorBase<ML4JTensorOperations> implements Tensor<ML4JTensorOperations> {

	private MatrixFactory matrixFactory;

	public ML4JTensor(Torch<ML4JTensorOperations> torch, MatrixFactory matrixFactory,
					  String name, String inputName, ML4JTensorOperations tensorOperations) {
		super(torch, name, inputName, tensorOperations);
		this.matrixFactory = matrixFactory;
	}

	public ML4JTensor(Torch<ML4JTensorOperations> torch, String name, String inputName,
					  ML4JTensorOperations ml4jTensorOperations) {
		super(torch, name, inputName, ml4jTensorOperations);
	}

	public ML4JTensor(Torch<ML4JTensorOperations> torch, MatrixFactory matrixFactory, String name, String inputName, Matrix matrix) {
		super(torch, name, inputName, new ML4JTensorOperations(matrixFactory, matrix));
		this.matrixFactory = matrixFactory;
	}

	public ML4JTensor(Torch<ML4JTensorOperations> torch, MatrixFactory matrixFactory, SymbolicTensor<ML4JTensorOperations> symbolicTensor) {
		super(torch, symbolicTensor);
		this.matrixFactory = matrixFactory;
	}

	@Override
	protected Tensor<ML4JTensorOperations> createDefaultTensor(Torch<ML4JTensorOperations> torch, SymbolicTensor<ML4JTensorOperations> tensor) {
		return new ML4JTensor(torch, matrixFactory, tensor);
	}

	@Override
	protected ML4JTensorOperations fromTensor(Tensor<ML4JTensorOperations> other) {
		
		if(other instanceof ML4JTensor){
            return ((ML4JTensor)other).symbolicTensor.get();
        } else {
        	if (other.size().getComponents().length != 2) {
    			throw new IllegalArgumentException("More than 2 dimensions");
    		}
    		Matrix matrix = matrixFactory.createMatrixFromRowsByRowsArray(other.size().get(0), 
    				other.size().get(1), other.getDataAsFloatArray());
    		return new ML4JTensorOperations(matrixFactory, matrix);
        }
	}

	@Override
	public Tensor<ML4JTensorOperations> view(int i, int j) {

		if (i == -1 && j == -1) {
			throw new RuntimeException("only one dimension can be inferred");
		} else {
			if (i == -1) {
				i = this.numel() / j;
			}
			if (j == -1) {
				j = this.numel() / i;
			}
		}

		Matrix view = toTensorOperations().getMatrix().softDup();
		view.asEditableMatrix().reshape(i, j);

		return this.performUnaryMappingOperation("view", new TensorOperationImpl<>("view", l -> new ML4JTensorOperations(matrixFactory, view), new Size (i, j)), new TensorOperationImpl<>("viewBackward", l -> l.view(size().get(0), size().get(1)), size()));

	}


	@Override
	public String toString() {

		List<String> lists = new ArrayList<>();
		Matrix matrix = symbolicTensor.evaluate().getMatrix();
		for (int r = 0; r < Math.min(2, matrix.getRows()); r++) {
			List<String> vals = new ArrayList<>();
			for (int c = 0; c < Math.min(matrix.getColumns(), 2); c++) {
				vals.add(Float.valueOf(matrix.get(r, c)).toString());
			}
			if (matrix.getColumns() > 2) {
				vals.add("...");
			}
			lists.add(vals.toString());
		}
		if (matrix.getRows() > 2) {
			lists.add("...");
		}

		if (requires_grad() && grad_fn() == null) {
			return "tensor(" + lists.toString() + ", requires_grad=True)";
		} else if (grad_fn()  != null) {
			return "tensor(name='" + this.symbolicTensor.getName() + "',"  + lists.toString() + ", grad_fn=" + grad_fn().toString() + ")";
		}
		else {
			return "tensor(name='" + this.symbolicTensor.getName()+ "',"  + lists.toString() + ")";
		}
	}

	@Override
	public Tensor<ML4JTensorOperations> self() {
		return this;
	}
}
