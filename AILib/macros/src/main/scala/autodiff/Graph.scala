package autodiff

import tensors.{Scalar, Tensor}
import tensors.TensorWrapper.*

// Note that strange behaviour might be caused when ConstantDoubles are involved, as well as rare cases with Sum
// in those cases a Scalar is the value of dx, while a high dimensional Tensor should be used
// => try to avoid using ConstantDoubles
trait Graph {
  /**
   * storage area for this nodes derivative
   */
  var dx: Tensor = _

  def compute(): Tensor

  def derive(): Unit
}

case class UnaryNode(op: UnaryOperator, g: Graph) extends Graph {
  override def compute(): Tensor = op.compute(g)

  override def derive(): Unit = op.derive(this, g)
}

case class BinaryNode(op: BinaryOperator, g1: Graph, g2: Graph) extends Graph {
  override def compute(): Tensor = op.compute(g1, g2)

  override def derive(): Unit = op.derive(this, g1, g2)
}

case class TrinaryNode(op: TrinaryOperator, g1: Graph, g2: Graph, g3: Graph) extends Graph {
  override def compute(): Tensor = op.compute(g1, g2, g3)

  override def derive(): Unit = op.derive(this, g1, g2, g3)
}

case class ParameterLeaf(name: String) extends Graph {
  var value: Tensor = _

  def setValue(tensor: Tensor): Unit = value = tensor

  override def compute(): Tensor = value

  override def derive(): Unit = {
    if (value.hasGrad) {
      value.getGrad.addGradient(dx)
    }
    if (value.hasGraph) {
      val subGraph = value.getGraph
      subGraph.dx = dx
      value.getGraph.derive()
    }
  }

}

case class Leaf(tensorID: Int, product: Product) extends Graph {
  def getValue: Any = product.productElement(tensorID)

  override def compute(): Tensor = getValue.asInstanceOf[Tensor]

  override def derive(): Unit = {
    val t = getValue.asInstanceOf[Tensor]
    if (t.hasGrad) {
      t.getGrad.addGradient(dx)
    }
  }
}

case class ConstantDouble(value: Double) extends Graph {
  override def compute(): Tensor = Scalar(value)

  override def derive(): Unit = ()
}

case object Empty extends Graph {
  override def compute(): Tensor = null

  override def derive(): Unit = ()
}


object Graph {
  import scala.quoted.*

  def apply(
             params: List[Expr[Graph]],
             operator: String,
             prodExpr: Expr[Product])(using Quotes): Expr[Graph] = {

    params.size match {
      case 1 => '{
      UnaryNode(
        UnaryOperator(${Expr(operator)}, ${prodExpr}),
        ${params.head}
      )
      }
      case 2 => '{
      BinaryNode(
        BinaryOperator(${Expr(operator)}, ${prodExpr}),
        ${params.head},
        ${params(1)}
      )
      }
      case 3 => '{
      TrinaryNode(
        TrinaryOperator(${Expr(operator)}, ${prodExpr}),
        ${params.head},
        ${params(1)},
        ${params(2)}
      )
      }
      case _ => ???
    }
  }

  def backwards(graph: Graph): Unit = {
    graph.dx = graph.compute().map(_ => 1)
    graph.derive()
  }
}
