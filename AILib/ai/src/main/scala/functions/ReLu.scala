package functions

import autodiff.Graph
import tensors.Tensor
import utils.ActivationFunction

case object ReLu extends ActivationFunction {
  override def activate(t: Tensor): Tensor = t.map(x => math.max(0.0, x))

  override def derive(t: Tensor): Tensor = t.map(x => if (x > 0) 1 else 0)
}
