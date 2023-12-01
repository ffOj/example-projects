package tensors

import Tensor.*
import utils.TensorOperators.*

case class Scalar(shape: OneD, initElements: Array[Double]) extends Tensor(shape, initElements) {

  override def toMe(doubles: Array[Double]): Tensor = Scalar(doubles.head)

  override def dot(t2: Tensor): Tensor = elements.head * t2

  override def transpose: Tensor = this
}

object Scalar {
  def apply(element: Double): Scalar = Scalar(OneD(1), Array[Double](element))
}