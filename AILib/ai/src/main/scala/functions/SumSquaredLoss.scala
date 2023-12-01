package functions

import tensors.Tensor
import utils.RecorderFunctions.*

import autodiff.record

case class SumSquaredLoss() extends CostFunction {
  override lazy val calculate = record(
    (y: Tensor, yHat: Tensor) => sum(square(minus(y, yHat))),
    this
  )
}
