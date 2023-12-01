package tensors

import autodiff.{Autograd, Graph}

import scala.annotation.tailrec

// indices for elements created by nesting for loops, with the added dimension being the innermost:
// e.g. row, column, depth, ...
// same as in truth tables
trait Tensor(val dimension: Tensor.Dimension, var elements: Array[Double]) {

  def toMe(doubles: Array[Double]): Tensor

  def elementWithIndices(indices: Int*): Double = elementWithIndices(indices.toList)

  def elementWithIndices(indices: List[Int]): Double = elements(findIndex(indices, dimension.asList))

  def setWithIndices(to: Double, indices: Int*): Unit = setWithIndices(to, indices.toList)

  def setWithIndices(to: Double, indices: List[Int]): Unit = elements(findIndex(indices, dimension.asList)) = to

  def indexOf(indices: Int*): Int = findIndex(indices.toList, dimension.asList)

  def indexOf(indices: List[Int]): Int = findIndex(indices, dimension.asList)

  private def findIndex(indices: List[Int], dimensions: List[Int]): Int = (indices, dimensions) match {
    case (idx :: Nil, dim :: Nil) => idx % dim
    case (idx :: tailIdx, dim :: tailDim) => (idx % dim) * tailDim.product + findIndex(tailIdx, tailDim)
    case _ => -1 // should not occur
  }

  def map(f: Double => Double): Tensor = toMe(elements.map(f))

  def zippedMap(t2: Tensor)(f: ((Double, Double)) => Double): Tensor = {
    if (dimension != t2.dimension) {
      val t = Tensor.broadcast(this, t2)
      t._1.zippedMap(t._2)(f)
    } else toMe(elements.zip(t2.elements).map(f))
  }

  def plus(t2: Tensor): Tensor = zippedMap(t2)(_ + _)

  def minus(t2: Tensor): Tensor = zippedMap(t2)(_ - _)

  def times(t2: Tensor): Tensor = zippedMap(t2)(_ * _)

  def divides(t2: Tensor): Tensor = zippedMap(t2)(_ / _)


  def plus(d: Double): Tensor = map(_ + d)

  def minus(d: Double): Tensor = map(_ - d)

  def times(d: Double): Tensor = map(_ * d)

  def divides(d: Double): Tensor = map(_ / d)

  // TODO: dot between Constant/Scalar and vector
  def dot(t2: Tensor): Tensor

  def transpose: Tensor
  
  // TODO: introduce axis
  def sum: Double = elements.sum

  def identity: Tensor = toMe(elements.map(_ => 1))

  protected[tensors] var grad: Option[Autograd] = None

  protected[tensors] var graph: Option[Graph] = None
}

object Tensor {
  trait Dimension {
    val asList: List[Int]
  }

  case class OneD(size: Int) extends Dimension {
    override val asList: List[Int] = List(size,1)
  }

  case class OneDT(size: Int) extends Dimension {
    override val asList: List[Int] = List(1,size)
  }

  case class TwoD(rows: Int, columns: Int) extends Dimension {
    override val asList: List[Int] = List(rows, columns)
  }

  case class ThreeD(rows: Int, columns: Int, depth: Int) extends Dimension {
    override val asList: List[Int] = List(rows, columns, depth)
  }

  case class FourD(rows: Int, columns: Int, depth: Int, trength: Int) extends Dimension {
    override val asList: List[Int] = List(rows, columns, depth, trength)
  }

  object Dimension {

    def apply(indices: Int*): Dimension = Dimension(indices.toList)

    def apply(indices: List[Int]): Dimension = indices.length match {
      case 1 => OneD(indices.head)
      case 2 if indices.head == 1 && indices.last == 1 => TwoD(1, 1)
      case 2 if indices.head == 1 => OneDT(indices(1))
      case 2 if indices.last == 1 => OneD (indices.head)
      case 2 => TwoD(indices.head, indices(1))
      case 3 => ThreeD(indices.head, indices(1), indices(2))
      case 4 => FourD(indices.head, indices(1), indices(2), indices(3))
      case _ => ???
    }
  }

  def apply(dimension: Dimension, elements: Array[Double]): Tensor = dimension match {
    case o: OneD => CVector(o, elements)
    case ot: OneDT => RVector(ot, elements)
    case t: TwoD => Matrix2D(t, elements)
    case t: ThreeD => Matrix3D(t, elements)
    case f: FourD => Matrix4D(f, elements)
    case _ => ???
  }

  def broadcast(t1: Tensor, t2: Tensor): (Tensor, Tensor) = {
    @tailrec def broadcastLists(l1: List[Int], l2: List[Int]): (List[Int], List[Int]) = (l1.length, l2.length) match {
      case (x1, x2) if x1 > x2 => broadcastLists(l1, l2 :+ 1)
      case (x1, x2) if x1 < x2 => broadcastLists(l1 :+ 1, l2)
      case _ => (l1, l2)
    }

    // step up existing dimensions
    def stepUp(t: Tensor, newDimensions: List[Int]): Array[_] = {
      // recurse through all positions in the new array and create the resulting multidimensional array
      def stepUp_(newDimensions : List[Int], currentIndices: List[Int]): Array[_] = {
        newDimensions match {
          // base case: make an array of all elements using the accumulated indices
          case head::Nil =>
            (for (i <- 0 until head) yield t.elementWithIndices(currentIndices :+ i)).toArray[Double]
          // step case: make an array of the arrays obtained by accumulating indices through recursion
          case head::tail =>
            val array = new Array[Array[_]](head)
            for (i <- 0 until head) array(i) = stepUp_(tail, currentIndices :+ i)
            array
          case Nil => ??? // should never occur - the tensor would be empty
        }
      }
      stepUp_(newDimensions, Nil)
    }

    // copies the current array to the higher dimensions
    @tailrec def copyToHigherDimensions(array: Array[_], toDimensions: List[Int]): Array[_] = toDimensions match {
      // base case: high dimensional array
      case Nil => array
      // step case: clone the current array head-times to obtain the next higher array and continue with next dimension
      case head::tail =>
        val resultingArray = new Array[Array[_]](head)
        for (i <- 0 until head) resultingArray(i) = array.clone()
        copyToHigherDimensions(resultingArray, tail)
    }

    // flattens a nested array of doubles
    def flattenNestedArray(array: Array[_]): Array[Double] = array match {
      case a: Array[Object] => a.map(x => flattenNestedArray(x.asInstanceOf[Array[_]])).flatten
      case a:Array[Double] => a
    }

    // equalise shapes of dimension-lists
    val lists = broadcastLists(t1.dimension.asList, t2.dimension.asList)
    // resulting shape is the maximum of each dimension length
    val resultingShape = lists._1.zip(lists._2).map(math.max(_, _))

    val resultingArrays = for (t <- List(t1, t2)) yield {
      val originalLength = t.dimension.asList.length
      flattenNestedArray(
        copyToHigherDimensions(
          stepUp(
            t,
            resultingShape.slice(0, originalLength)
          ),
          resultingShape.slice(originalLength, resultingShape.length)
        )
      )
    }

    import Tensor.Dimension
    (
      Tensor(Dimension(resultingShape), resultingArrays.head),
      Tensor(Dimension(resultingShape), resultingArrays(1))
    )
  }
}
