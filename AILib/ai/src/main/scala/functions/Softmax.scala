package functions

import autodiff.Graph
import tensors.Tensor
import utils.ActivationFunction
import utils.TensorOperators.*

case object Softmax extends ActivationFunction {
  
  override def activate(t: Tensor): Tensor = {
    val sums = t.elements.map(math.exp).sum
    val newElements = t.elements.map(
      math.exp(_) / sums
    )
    Tensor(t.dimension, newElements)
  }
  
  override def derive(t: Tensor): Tensor = activate(t) * (1 - activate(t))
}
