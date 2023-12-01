import tensors.*
import tensors.Tensor.ThreeD

object Reader {

  def oneHotOfDigit(digit: Int): CVector = {
    val vector = CVector(10, new Array[Double](10))
    vector.elements(digit) = 1
    vector
  }

  def readMNIST(fileName: String): List[(Matrix3D, CVector)] = {
    def readMNIST_(lines: List[String],
                   ret: List[(Matrix3D, CVector)]
                  ): List[(Matrix3D, CVector)] = lines match {
      case Nil => ret
      case head :: tail => {
        val list = head.split(",")
        val matrix = Matrix3D(ThreeD(28,28,1), new Array[Double](28*28))
        for (r <- 0 until 28) {
          for (c <- 0 until 28) {
            matrix.setWithIndices(list((r * 28) + c + 1).toDouble / 255, r, c, 0)
          }
        }
        readMNIST_(tail, (matrix, oneHotOfDigit(list.head.toInt)) :: ret)
      }
    }

    readMNIST_(scala.io.Source.fromFile(fileName).getLines().toList, Nil)
  }
}
