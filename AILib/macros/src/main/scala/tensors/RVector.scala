package tensors

import Tensor.*
import utils.TensorOperators.*

case class RVector(shape: OneDT, initElements: Array[Double]) extends Tensor(shape, initElements) {
  override def toMe(doubles: Array[Double]): Tensor = RVector(shape, doubles)

  override def dot(t2: Tensor): Tensor = t2 match {
    case c: CVector => Scalar((this * c.T).sum)
    case m: Matrix2D => dot(m)
    case _ => ???
  }

  def dot(m: Matrix2D): Tensor = {
    val result: Array[Double] = new Array[Double](m.shape.columns)

    for (i <- 0 until m.shape.columns) {
      for (j <- 0 until shape.size) {
        val otherIdx: Int = j * m.shape.columns + i
        result(i) += elements(j) * m.elements(otherIdx)
      }
    }

    RVector(OneDT(m.shape.columns), result)
  }

  override def transpose: Tensor = CVector(OneD(shape.size), elements)
}

object RVector {
  def apply(size: Int, elements: Array[Double]): RVector = RVector(OneDT(size), elements)
  def apply(size: Int, elements: Double*): RVector = RVector(OneDT(size), elements.toArray)
}
