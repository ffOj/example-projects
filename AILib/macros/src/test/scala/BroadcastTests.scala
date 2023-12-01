import org.junit.Assert.*
import org.junit.Test

import tensors.*
import tensors.Tensor.*

class BroadcastTests:

  // testing on 2x2, 3x1 tensors
  @Test def MatrixCVectorBroadcasting(): Unit = {
    val M = Matrix2D(
      rows = 2, columns = 2,
      1,2,
      3,4
    )

    val v = CVector(
      size = 3,
      1,2,3
    )

    val result = Tensor.broadcast(M, v)
    // expectation: 1 2   1        1 2   1 1
    //              3 4 , 2   =>   3 4 , 2 2
    //                    3        1 2   3 3
    assertTrue(result._1.dimension == result._2.dimension && result._1.dimension == TwoD(3,2))
    assertTrue(result._1.elements.toList == List(1,2,3,4,1,2) && result._2.elements.toList == List(1,1,2,2,3,3))
  }

  // testing on 2x2, 1x1 tensors
  @Test def MatrixCVectorBroadcasting2(): Unit = {
    val M = Matrix2D(
      rows = 2, columns = 2,
      1,2,
      3,4
    )

    val v = CVector(
      size = 1,
      1
    )

    val result = Tensor.broadcast(M, v)
    // expectation: 1 2   1        1 2  1 1
    //              3 4 ,     =>   3 4, 1 1
    assertTrue(result._1.dimension == result._2.dimension && result._1.dimension == TwoD(2,2))
    assertTrue(result._1.elements.toList == List(1,2,3,4) && result._2.elements.toList == List(1,1,1,1))
  }

  // testing on 4x2, 3x1 tensors
  @Test def MatrixCVectorBroadcasting3(): Unit = {
    val M = Matrix2D(
      rows = 4, columns = 2,
      1,2,
      3,4,
      5,6,
      7,8
    )

    val v = CVector(
      size = 3,
      1,2,3
    )

    val result = Tensor.broadcast(M, v)
    // expectation: 1 2   1        1 2   1 1
    //              3 4   2   =>   3 4   2 2
    //              5 6   3        5 6   3 3
    //              7 8 ,          7 8 , 1 1
    assertTrue(result._1.dimension == result._2.dimension && result._1.dimension == TwoD(4,2))
    assertTrue(result._1.elements.toList == List(1,2,3,4,5,6,7,8) && result._2.elements.toList == List(1,1,2,2,3,3,1,1))
  }

  // testing on 2x2, 1x3 tensors
  @Test def MatrixRVectorBroadcasting(): Unit = {
    val M = Matrix2D(
      rows = 2, columns = 2,
      1,2,
      3,4
    )

    val v = RVector(
      size = 3,
      1,2,3
    )

    val result = Tensor.broadcast(M, v)
    // expectation: 1 2                1 2 1   1 2 3
    //              3 4 , 1 2 3   =>   3 4 3 , 1 2 3
    assertTrue(result._1.dimension == result._2.dimension && result._1.dimension == TwoD(2,3))
    assertTrue(result._1.elements.toList == List(1,2,1,3,4,3) && result._2.elements.toList == List(1,2,3,1,2,3))
  }

  // testing on 2x2, 3x3 tensors
  @Test def MatrixMatrixBroadcasting(): Unit = {
    val M1 = Matrix2D(
      2,2,
      1,2,
      3,4
    )

    val M2 = Matrix2D(
      3,3,
      1,2,3,
      4,5,6,
      7,8,9
    )

    val result = Tensor.broadcast(M1, M2)
    // expectation: 1 2   1 2 3        1 2 1   1 2 3
    //              3 4 , 4 5 6   =>   3 4 3   4 5 6
    //                    7 8 9        1 2 1 , 7 8 9
    assertTrue(result._1.dimension == result._2.dimension && result._1.dimension == TwoD(3,3))
    assertTrue(
      result._1.elements.toList == List(1,2,1,3,4,3,1,2,1) && result._2.elements.toList == List(1,2,3,4,5,6,7,8,9)
    )
  }
