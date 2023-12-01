import org.junit.Assert.*
import org.junit.Test
import org.junit.Before

import utils.TensorOperators.*
import utils.RecorderFunctions.*
import layers.Dense
import tensors.TensorWrapper.*
import functions.Sigmoid
import tensors.*

import autodiff.*

case class ComputationalGraphTests() {

  var cv: Option[CVector] = None
  var rv: Option[RVector] = None
  var M: Option[Matrix2D] = None
  var denseLayer: Option[Dense] = None

  case class Linear(w: Matrix2D, b: CVector) {
    lazy val f: Tensor => Tensor = record(
      (x: Tensor) =>
        plus(
          dot(w, x),
          b
        ),
      this
    )
  }

  @Before def initialize(): Unit = {
    cv = Option(
      CVector(
        4,
        1,2,3,4
      )
    )

    rv = Option(
      RVector(
        4,
        5,6,7,8
      )
    )

    M = Option(
      Matrix2D(
        (4, 4),
        1,2,3,4,
        5,6,7,8,
        1,2,3,4,
        5,6,7,8
      )
    )

    denseLayer = Option(
      Dense(M.get, cv.get, Sigmoid)
    )
  }

  @Test def binaryWithConstGraph(): Unit = {
    case class F1() {
      lazy val f: Tensor => Tensor = record(
        (x: Tensor) => plus(1, x),
        this
      )
    }
    lazy val res = F1().f(cv.get)

    assertTrue(res.getGraph == BinaryNode(Addition, ConstantDouble(1), ParameterLeaf("x")))
  }

  @Test def binaryWithSelfGraph(): Unit = {
    case class F2() {
      lazy val f: Tensor => Tensor = record(
        (x: Tensor) => plus(x, x),
        this
      )
    }
    val res = F2().f(cv.get.asInstanceOf[Tensor])

    println(res.getGraph)
    assertTrue(res.getGraph == BinaryNode(Addition, ParameterLeaf("x"), ParameterLeaf("x")))
  }

  @Test def binaryWithOtherGraph(): Unit = {
    val x = CVector(cv.get.shape, cv.get.initElements)

    case class CLS(cv: CVector) {
      lazy val f: Tensor => Tensor = record(
        x => plus(x, cv),
        this
      )
    }
    val cls = CLS(x)
    val res = cls.f(cv.get)

    assertTrue(res.getGraph == BinaryNode(Addition, ParameterLeaf("x"), Leaf(0, cls)))
  }

  @Test def unaryGraph(): Unit = {
    case class F3() {
      lazy val f: Tensor => Tensor = record(
        x => square(x),
        this
      )
    }

    val res = F3().f(cv.get)

    assertTrue(res.getGraph == UnaryNode(Square, ParameterLeaf("x")))
  }

  @Test def denseGraph(): Unit = {
    val res = denseLayer.get.predict(cv.get)

    val expected: Graph = UnaryNode(
      Activation("activation", denseLayer.get),
      BinaryNode(
        Addition,
        BinaryNode(
          Dot,
          Leaf(0, denseLayer.get),
          ParameterLeaf("x")
        ),
        Leaf(1, denseLayer.get)
      )
    )

    assertTrue(res.getGraph == expected)
  }

  @Test def nestedLinearForwardComputation(): Unit = {
    val l1 = Linear(M.get, cv.get)
    val l2 = Linear(M.get, cv.get)
    val res = l1.f(l2.f(cv.get))
    val forwardComputation = res.getGraph.compute()
    val expected =  M.get `@` (M.get `@` cv.get + cv.get) + cv.get

    assertTrue(res.dimension == forwardComputation.dimension && res.elements.toList == forwardComputation.elements.toList)
    assertTrue(res.dimension == expected.dimension && res.elements.toList == expected.elements.toList)
  }

  @Test def denseDerivative(): Unit = {
    M.get.attachGrad()
    val res = denseLayer.get.predict(cv.get)
    val expected = Sigmoid.derive(M.get `@` cv.get + cv.get) * (cv.get `@` cv.get.identity.T)

    res.backwards()
    val gradient: Tensor = M.get.getGrad.getGradient

    assertTrue(gradient.dimension == expected.dimension && gradient.elements.toList == expected.elements.toList)
    M.get.removeGrad()
  }

  @Test def nestedLinearLayersDerivative(): Unit = {
    val M2 = Matrix2D(
      (4, 4),
      1,2,3,4,
      2,3,4,5,
      3,4,5,6,
      4,5,6,7
    )
    val cv2 = CVector(
      4,
      1,2,3,4
    )

    M2.attachGrad()

    val l1 = Linear(M.get, cv.get)
    val l2 = Linear(M2, cv2)
    val intermediate = l2.f(cv.get)
    val res = l1.f(intermediate)

    res.backwards()

    val gradient = M2.getGrad.getGradient
    val expected = M.get `@` (cv.get `@` cv.get.identity.T)

    assertTrue(gradient.dimension == expected.dimension && gradient.elements.toList == expected.elements.toList)
    M2.removeGrad()
  }

  @Test def nestedDenseDerivative(): Unit = {
    val M2 = Matrix2D(
      (4,4),
      1,2,3,4,
      2,3,4,5,
      3,4,5,6,
      4,5,6,7
    )

    val cv2 = CVector(
      4, 1,2,3,4
    )
    M2.attachGrad()
    M.get.attachGrad()

    val denseLayer2 = Dense(M2, cv2, Sigmoid)

    val res = denseLayer.get.predict(denseLayer2.predict(cv.get))
    res.backwards()

    var gradient = M2.getGrad.getGradient

    val inner = M2 `@` cv.get + cv2
    var expected = Sigmoid.derive(M.get `@` Sigmoid.activate(inner) + cv.get) * M.get `@` (Sigmoid.derive(inner) * (cv.get `@` cv.get.identity.T))

    assertTrue(gradient.dimension == expected.dimension && gradient.elements.toList == expected.elements.toList)

    gradient = M.get.getGrad.getGradient
    expected = Sigmoid.derive(M.get `@` Sigmoid.activate(inner) + cv.get) * (Sigmoid.activate(inner) `@` Sigmoid.activate(inner).identity.T)

    assertTrue(gradient.dimension == expected.dimension && gradient.elements.toList == expected.elements.toList)
    
    M.get.removeGrad()
    M2.removeGrad()
  }
}