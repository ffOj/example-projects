package layers

import tensors.Tensor
import utils.RecorderFunctions.*

case class AvgPooling(windowSize: (Int, Int), stride: (Int, Int), id: Int) extends Layer {
  override def predict(input: Tensor): Tensor = predictFunction(input)

  private lazy val predictFunction = autodiff.record(
    (x: Tensor) => avgpool2D(x, windowSize, stride),
    this
  )
}
