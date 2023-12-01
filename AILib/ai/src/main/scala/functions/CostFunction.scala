package functions

import tensors.Tensor

trait CostFunction {
  lazy val calculate: (Tensor, Tensor) => Tensor
}
