package tensors

import autodiff.{Autograd, Graph}

object TensorWrapper {
  extension (t: Tensor) {
    def attachGrad(): Unit = t.grad = Option(Autograd())

    def removeGrad(): Unit = t.grad = None

    def hasGrad: Boolean = t.grad.isDefined

    def getGrad: Autograd = t.grad.get

    def setGraph(g: Graph): Unit = t.graph = Option(g)

    def hasGraph: Boolean = t.graph.isDefined

    def getGraph: Graph = t.graph.get

    def backwards(): Unit = Graph.backwards(getGraph)
  }
}
