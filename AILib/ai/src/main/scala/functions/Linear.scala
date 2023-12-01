package functions

import autodiff.Graph
import tensors.Tensor
import utils.ActivationFunction

case object Linear extends ActivationFunction {
  override def activate(t: Tensor): Tensor = t

  override def derive(t: Tensor): Tensor = t.map(_ => 1)
}