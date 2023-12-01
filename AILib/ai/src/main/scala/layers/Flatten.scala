package layers

import tensors.Tensor
import utils.RecorderFunctions.flatten
import autodiff.record

case class Flatten(fromSize: List[Int]) extends Layer {
  override def predict(input: Tensor): Tensor = predictFunction(input)

  private lazy val predictFunction: Tensor => Tensor = record(
    (input: Tensor) => flatten(input),
    this
  )
}
