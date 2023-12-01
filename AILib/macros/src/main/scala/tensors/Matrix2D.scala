package tensors

import Tensor.*

case class Matrix2D(shape:TwoD, initElements: Array[Double]) extends Tensor(shape, initElements) {
  override def toMe(doubles:Array[Double]): Tensor = Matrix2D(shape, doubles)

  override def dot(t2: Tensor): Tensor = t2 match {
    case c: CVector => dot(c)
    case m: Matrix2D => dot(m)
    case _ =>
      println((elements.toList, t2.elements.toList, shape, t2.dimension))
      ???
  }

  def dot(c:CVector): CVector = {
    val result: Array[Double] = new Array[Double](shape.rows)

    for (i <- 0 until shape.rows) {
      for (j <- 0 until c.shape.size) {
        result(i) += this.elements(i * c.shape.size + j) * c.elements(j)
      }
    }

    CVector(OneD(shape.rows), result)
  }

  def dot(m: Matrix2D): Matrix2D = {
    val result: Array[Double] = new Array[Double](shape.rows * m.shape.columns)

    for (k <- 0 until shape.rows) {
      for (i <- 0 until m.shape.columns) {
        for (j <- 0 until shape.columns) {
          val thisIdx: Int = k * shape.columns + j
          val otherIdx: Int = j * m.shape.columns + i
          val resIdx: Int = k * m.shape.columns + i
          result(resIdx) += elements(thisIdx) * m.elements(otherIdx)
        }
      }
    }

    Matrix2D(TwoD(shape.rows, m.shape.columns), result)
  }

  override def transpose: Tensor = {
    val result: Array[Double] = new Array[Double](elements.length)

    for (i <- 0 until shape.rows) {
      for (j <- 0 until shape.columns) {
        val resIdx: Int = j * shape.rows + i
        val thisIdx: Int = i * shape.columns + j
        result(resIdx) = elements(thisIdx)
      }
    }

    Matrix2D(TwoD(shape.columns, shape.rows), result)
  }

}

object Matrix2D {
  def apply(rows: Int, columns: Int, elements: Array[Double]): Matrix2D = Matrix2D(TwoD(rows, columns), elements)

  def apply(rows: Int, columns: Int, elements: Double*): Matrix2D = Matrix2D(TwoD(rows, columns), elements.toArray)

  def apply(shape: (Int, Int), elements: Double*): Matrix2D = Matrix2D(TwoD(shape._1, shape._2), elements.toArray)

  def apply(shape: (Int, Int), elements: Array[Double]): Matrix2D = Matrix2D(TwoD(shape._1, shape._2), elements)
}
