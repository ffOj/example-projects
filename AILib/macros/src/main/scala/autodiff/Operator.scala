package autodiff

import tensors.{Matrix3D, Scalar, Tensor}
import utils.TensorOperators.*
import utils.ActivationFunction
import utils.RecorderFunctions.*

trait UnaryOperator {
  def compute(g: Graph): Tensor

  def derive(node: Graph, g: Graph): Unit
}

trait BinaryOperator {
  def compute(g1: Graph, g2: Graph): Tensor

  def derive(node: Graph, g1: Graph, g2: Graph): Unit
}

trait TrinaryOperator {
  def compute(g1: Graph, g2: Graph, g3: Graph): Tensor

  def derive(node: Graph, g1: Graph, g2: Graph, g3: Graph): Unit
}


case object Addition extends BinaryOperator {
  override def compute(g1: Graph, g2: Graph): Tensor = g1.compute() + g2.compute()

  override def derive(node: Graph, g1: Graph, g2: Graph): Unit = {
    g1.dx = node.dx
    g2.dx = node.dx
    g1.derive()
    g2.derive()
  }
}

case object Subtraction extends BinaryOperator {
  override def compute(g1: Graph, g2: Graph): Tensor = g1.compute() - g2.compute()

  override def derive(node: Graph, g1: Graph, g2: Graph): Unit = {
    g1.dx = node.dx
    g2.dx = 0 - node.dx
    g1.derive()
    g2.derive()
  }
}

case object Multiplication extends BinaryOperator {
  override def compute(g1: Graph, g2: Graph): Tensor = g1.compute() * g2.compute()

  override def derive(node: Graph, g1: Graph, g2: Graph): Unit = {
    g1.dx = g2.compute() * node.dx
    g2.dx = g1.compute() * node.dx
    g1.derive()
    g2.derive()
  }
}

case object Dot extends BinaryOperator {
  override def compute(g1: Graph, g2: Graph): Tensor = g1.compute() `@` g2.compute()

  override def derive(node: Graph, g1: Graph, g2: Graph): Unit = {
    g1.dx = node.dx `@` g2.compute().T
    g2.dx = g1.compute().T `@` node.dx
    g1.derive()
    g2.derive()
  }
}

case class Activation(id: String, product: Product) extends UnaryOperator {
  val idIndex: Int = product.productElementNames.indexOf(id)

  def getFunction: ActivationFunction = product.productElement(idIndex).asInstanceOf[ActivationFunction]

  override def compute(g: Graph): Tensor = getFunction.activate(g.compute())

  override def derive(node: Graph, g: Graph): Unit = {
    g.dx = node.dx * getFunction.derive(g.compute())
    g.derive()
  }
}

case object Sum extends UnaryOperator {
  override def compute(g: Graph): Tensor = Scalar(g.compute().sum)

  override def derive(node: Graph, g: Graph): Unit = {
    g.dx = g.compute().map(_ => node.dx.elements.head)
    g.derive()
  }
}

case object Transpose extends UnaryOperator {
  override def compute(g: Graph): Tensor = g.compute().T

  override def derive(node: Graph, g: Graph): Unit = {
    g.dx = node.dx.T
    g.derive()
  }
}

case object Square extends UnaryOperator {
  override def compute(g: Graph): Tensor = g.compute().map(math.pow(_, 2))

  override def derive(node: Graph, g: Graph): Unit = {
    g.dx = node.dx * 2 * g.compute()
    g.derive()
  }
}

case object Log extends UnaryOperator {
  override def compute(g: Graph): Tensor = g.compute().map(math.log)

  override def derive(node: Graph, g: Graph): Unit = {
    g.dx = node.dx / g.compute()
    g.derive()
  }
}

case class Flatten(product: Product) extends UnaryOperator {
  // can be obtained from g.compute()
  private val fromSize = product.productElement(product.productElementNames.indexOf("fromSize")).asInstanceOf[List[Int]]

  override def compute(g: Graph): Tensor = flatten(g.compute())

  override def derive(node: Graph, g: Graph): Unit = {
    g.dx = Tensor(
      Tensor.Dimension(fromSize),
      node.dx.elements
    )
    g.derive()
  }
}

case object Convolution extends TrinaryOperator {
  override def compute(g1: Graph, g2: Graph, g3: Graph): Tensor = {
    convolution2D(
      g1.compute(),
      g2.compute(),
      g3.asInstanceOf[Leaf].getValue.asInstanceOf[(Int, Int)]
    )
  }

  override def derive(node: Graph, g1: Graph, g2: Graph, g3: Graph): Unit = {
    val stride = g3.asInstanceOf[Leaf].getValue.asInstanceOf[(Int, Int)]
    g1.dx = convolution2D(node.dx, g2.compute(), stride) // weight gradient using convolution of input and error
    g2.dx = flippedFullConvolution2D(g1.compute(), node.dx, stride) // input gradient using full convolution of weights and error
    g1.derive()
    g2.derive()
  }
}

case object MaxPool2D extends TrinaryOperator {
  override def compute(g1: Graph, g2: Graph, g3: Graph): Tensor = maxpool2D(
    g1.compute(),
    g2.asInstanceOf[Leaf].getValue.asInstanceOf[(Int, Int)],
    g3.asInstanceOf[Leaf].getValue.asInstanceOf[(Int, Int)]
  )

  override def derive(node: Graph, g1: Graph, g2: Graph, g3: Graph): Unit = {
    val input = g1.compute()
    val indices = argmax(
      input,
      g2.asInstanceOf[Leaf].getValue.asInstanceOf[(Int, Int)],
      g3.asInstanceOf[Leaf].getValue.asInstanceOf[(Int, Int)]
    )
    g1.dx = Tensor(
      input.dimension,
      {
        val res = new Array[Double](input.dimension.asList.product)
        indices.elements.zip(node.dx.elements).foreach(
          (t: (Double, Double)) => {
            val idx = t._1.toInt
            res(idx) += t._2
          }
        )
        res
      }
    )
    g1.derive()
  }
}

case object AvgPool2D extends TrinaryOperator {
  override def compute(g1: Graph, g2: Graph, g3: Graph): Tensor = maxpool2D(
    g1.compute(),
    g2.asInstanceOf[Leaf].getValue.asInstanceOf[(Int, Int)],
    g3.asInstanceOf[Leaf].getValue.asInstanceOf[(Int, Int)]
  )

  override def derive(node: Graph, g1: Graph, g2: Graph, g3: Graph): Unit = {
    val input = g1.compute().asInstanceOf[Matrix3D]
    val windowSize = g2.asInstanceOf[Leaf].getValue.asInstanceOf[(Int, Int)]

    val rows = input.shape.rows
    val columns = input.shape.columns
    val depth = input.shape.depth

    val res = Matrix3D(input.shape, new Array[Double](input.elements.length))

    val window = windowSize._1 * windowSize._2
    for (r <- 0 until rows) {
      for (c <- 0 until columns) {
        for (d <- 0 until depth) {
          for (kr <- 0 until windowSize._1) {
            for (kc <- 0 until windowSize._2) {
              res.elements(res.indexOf(r, c, d)) += node.dx.elementWithIndices(kr, kc, d) / window
            }
          }
        }
      }
    }
    g1.dx = res
    g1.derive()
  }
}


object UnaryOperator {
  def apply(str: String, product: Product): UnaryOperator = {
    val l: List[String] = str.split('.').toList
    l match {
      case id :: "activate" :: Nil => Activation(id, product)
      case _ :: "sum" :: Nil => Sum
      case _ :: "square" :: Nil => Square
      case _ :: "log" :: Nil => Log
      case _ :: "transpose" :: Nil => Transpose
      case _ :: "flatten" :: Nil => Flatten(product)
      case _ => ???
    }
  }
}

object BinaryOperator {
  def apply(str: String, product: Product): BinaryOperator = {
    val l: List[String] = str.split('.').toList
    l match {
      case _ :: "plus" :: Nil => Addition
      case _ :: "times" :: Nil => Multiplication
      case _ :: "dot" :: Nil => Dot
      case _ :: "minus" :: Nil => Subtraction
      case _ => ???
    }
  }
}

object TrinaryOperator {
  def apply(str: String, product: Product): TrinaryOperator = {
    val l: List[String] = str.split('.').toList
    l match {
      case _ :: "convolution2D" :: Nil => Convolution
      case _ :: "maxpool2D" :: Nil => MaxPool2D
      case _ :: "avgpool2D" :: Nil => AvgPool2D
      case _ => ???
    }
  }
}

object Operator {
  var supportedOps: List[String] =
    List("plus", "minus", "times", "divides", "dot", "sum", "square", "log", "transpose",
      "convolution2D", "flatten", "maxpool2D", "avgpool2D")

}

