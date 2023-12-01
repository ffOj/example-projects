package tensors

import Tensor.*

case class CVector(shape: OneD, initElements: Array[Double]) extends Tensor(shape, initElements) {

  override def toMe(doubles: Array[Double]): Tensor = CVector(shape, doubles)

  override def dot(t2: Tensor): Tensor = t2 match {
    case r: RVector => dot(r)
    case _ => ???
  }

  def dot(r: RVector): Matrix2D = {
    val result: Array[Double] = new Array[Double](shape.size * r.shape.size)

    for (i <- 0 until shape.size) {
      for (j <- 0 until r.shape.size) {
        val idx = i * r.shape.size + j
        result(idx) = elements(i) * r.elements(j)
      }
    }

    Matrix2D(
      TwoD(shape.size, r.shape.size),
      result
    )
  }

  override def transpose: Tensor = RVector(OneDT(shape.size), elements)
}
object CVector {
  def apply(size: Int, elements: Array[Double]): CVector = CVector(OneD(size), elements)
  def apply(size: Int, elements: Double*): CVector = CVector(OneD(size), elements.toArray)
}
