package tensors

import Tensor.ThreeD

case class Matrix3D(shape: ThreeD, elements_ : Array[Double]) extends Tensor(shape, elements_) {
  override def toMe(doubles: Array[Double]): Tensor = Matrix3D(shape, doubles)

  override def transpose: Tensor = ???

  override def dot(t2: Tensor): Tensor = ???
  
  def flip180: Matrix3D = {
    val res = Matrix3D(shape, new Array[Double](elements.length))

    for (r <- 0 until shape.rows) {
      for (c <- 0 until shape.columns) {
        for (d <- 0 until shape.depth) {
          res.setWithIndices(
            super.elementWithIndices(r, c, d),
            shape.rows - r - 1,
            shape.columns - c - 1,
            d
          )
        }
      }
    }
    res
  }
}

object Matrix3D {
  def apply(rows: Int, columns: Int, depth: Int, elements: Double*): Matrix3D = {
    Matrix3D(ThreeD(rows, columns, depth), elements.toArray)
  }
}