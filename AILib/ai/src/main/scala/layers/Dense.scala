package layers

import tensors.*
import utils.ActivationFunction
import autodiff.record
import utils.RecorderFunctions.*
import utils.TensorOperators.*

import scala.util.Random


case class Dense(var weights: Matrix2D,
                 var biases: CVector,
                 activation: ActivationFunction) extends ParameterizedLayer[Matrix2D, CVector] {

  override def predict(input: Tensor): Tensor = predictFunction(input)

  private lazy val predictFunction = record(
    (x: Tensor) => activation.activate(
      plus(
        dot(weights, x),
        biases
      )
    ), this
  )
}

object Dense {
  val rng: Random = scala.util.Random()

  def apply(inputs: Int, outputs: Int, activation: ActivationFunction): Dense = {

    import tensors.TensorWrapper.*

    val w = Matrix2D(outputs, inputs, for (_ <- (0 until inputs*outputs).toArray) yield rng.between(-0.25, 0.25))
    val b = CVector(outputs, for (_ <- (0 until inputs*outputs).toArray) yield rng.between(-0.25, 0.25))
    w.attachGrad()
    b.attachGrad()

    Dense(weights = w, biases = b, activation)
  }
}
