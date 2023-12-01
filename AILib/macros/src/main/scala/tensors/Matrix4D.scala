package tensors

import Tensor.{ThreeD, FourD}

case class Matrix4D(shape: FourD, elements_ : Array[Double]) extends Tensor(shape, elements_) {
  
  override def toMe(doubles: Array[Double]): Tensor = Matrix4D(shape, doubles)

  override def transpose: Tensor = ???

  override def dot(t2: Tensor): Tensor = ???

  def toMatrix3Ds: List[Matrix3D] = {
    val matrixList: List[Matrix3D] =
      (
        for (_ <- 0 until shape.trength) yield Matrix3D(
          ThreeD(shape.rows, shape.columns, shape.depth),
          new Array[Double](shape.rows * shape.columns * shape.depth)
        )
      ).toList
    for (x <- 0 until shape.rows) {
      for (y <- 0 until shape.columns) {
        for (z <- 0 until shape.depth) {
          for (w <- 0 until shape.trength) {
            matrixList(w).setWithIndices(
              elementWithIndices(x, y, z, w), x, y, z
            )
          }
        }
      }
    }
    matrixList
  }
}
