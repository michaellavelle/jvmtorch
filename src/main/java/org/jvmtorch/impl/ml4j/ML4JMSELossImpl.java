package org.jvmtorch.impl.ml4j;

import org.jvmtorch.impl.MSELossImpl;
import org.jvmtorch.impl.TensorOperationImpl;
import org.jvmtorch.nn.modules.MSELoss;
import org.jvmtorch.torch.Size;
import org.jvmtorch.torch.Tensor;
import org.jvmtorch.torch.TensorConverter;
import org.jvmtorch.torch.TensorData;
import org.jvmtorch.torch.TensorDataConverter;
import org.jvmtorch.torch.Torch;
import org.ml4j.Matrix;
import org.ml4j.nn.activationfunctions.ActivationFunctionBaseType;
import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.costfunctions.CostFunction;
import org.ml4j.nn.costfunctions.DeltaRuleCostFunctionGradientImpl;
import org.ml4j.nn.costfunctions.SumSquaredErrorCostFunction;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.format.features.DimensionScope;

public class ML4JMSELossImpl extends MSELossImpl {

	private Torch torch;
	private CostFunction costFunction;
	private TensorDataConverter<ML4JTensorOperations> tensorDataConverter;
	private TensorConverter<ML4JTensor> tensorConverter;

	public ML4JMSELossImpl(Torch torch, TensorDataConverter<ML4JTensorOperations> tensorDataConverter,
			TensorConverter<ML4JTensor> tensorConverter) {
		this.torch = torch;
		this.costFunction = new SumSquaredErrorCostFunction();
		this.tensorDataConverter = tensorDataConverter;
		this.tensorConverter = tensorConverter;
	}

	@Override
	public Tensor forward(MSELoss self, Tensor input, Tensor target) {
		ML4JTensor ml4jTensor = tensorConverter.createTensor(input);

		ML4JTensor ml4jTargetTensor = tensorConverter.createTensor(target);

		NeuronsActivation inputNeuronsActivation = ml4jTensor.toNeuronsActivation(DimensionScope.INPUT,
				NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);

		NeuronsActivation targetNeuronsActivation = ml4jTargetTensor.toNeuronsActivation(DimensionScope.INPUT,
				NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);

		Matrix inputMatrix = inputNeuronsActivation
				.getActivations(ml4jTensor.getDirectedComponentsContext().getMatrixFactory());

		Matrix targetMatrix = targetNeuronsActivation
				.getActivations(ml4jTensor.getDirectedComponentsContext().getMatrixFactory());

		float cost = costFunction.getAverageCost(targetMatrix, inputMatrix);

		DeltaRuleCostFunctionGradientImpl costFunctionGradient = new DeltaRuleCostFunctionGradientImpl(
				ml4jTargetTensor.getDirectedComponentsContext().getMatrixFactory(), costFunction,
				targetNeuronsActivation, inputNeuronsActivation);

		DirectedComponentGradient<NeuronsActivation> gradient = costFunctionGradient
				.backPropagateThroughFinalActivationFunction(
						ActivationFunctionType.getBaseType(ActivationFunctionBaseType.LINEAR));

		NeuronsActivation output = gradient.getOutput();

		ML4JTensor out2 = new ML4JTensor(torch, ml4jTensor.getDirectedComponentsContext(), tensorDataConverter, output,
				false);

		Tensor outputTensor = input.performUnaryMappingOperation(
				new TensorOperationImpl<TensorData, Size>(torch, "LossOutput", l -> torch.tensor(cost).toTensorData(),
						s -> torch.Size()),
				new TensorOperationImpl<>(torch, "LossBackward", l -> out2.setCostFunctionGradient(true),
						s -> out2.size()));

		return outputTensor;
	}
}
