package functions

import tensors.Tensor
import utils.ActivationFunction
import utils.TensorOperators.*

case object Sigmoid extends ActivationFunction {
  override def activate(t: Tensor): Tensor = t.map((x: Double) => 1 / (1 + math.exp(-x)))

  override def derive(t: Tensor): Tensor = activate(t) * (1 - activate(t))
}
