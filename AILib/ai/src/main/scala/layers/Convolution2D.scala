package layers

import tensors.*
import tensors.Tensor.*
import utils.RecorderFunctions.*
import utils.ActivationFunction
import autodiff.record

import scala.util.Random

case class Convolution2D(var weights: Matrix4D,
                         var biases: Matrix3D,
                         stride: (Int, Int),
                         var activation: ActivationFunction) extends ParameterizedLayer[Matrix4D, Matrix3D] {

  override def predict(input: Tensor): Tensor = predictFunction(input)

  private lazy val predictFunction: Tensor => Tensor = record(
    input => activation.activate(
      plus(
        convolution2D(weights, input, stride),
        biases
      )
    ),
    this
  )

  override def update(learningRate: Double): Unit = {
    import tensors.TensorWrapper.*
    import utils.TensorOperators.*

    val wGrad = weights.getGrad.getGradient

    weights.elements = (weights - (learningRate * wGrad)).elements
  }
}

object Convolution2D {
  val rng: Random = scala.util.Random()

  def apply(filterRows: Int,
            filterColumns: Int,
            inputDepth: Int,
            numbFilters: Int,
            outputRows: Int,
            outputColumns: Int,
            stride: (Int, Int),
            activation: ActivationFunction): Convolution2D = {

    import tensors.TensorWrapper.*

    val w = Matrix4D(
      FourD(filterRows, filterColumns, inputDepth, numbFilters),
      for (_ <- (0 until filterRows * filterColumns * inputDepth * numbFilters).toArray) yield rng.between(-0.25, 0.25)
    )
    val b = Matrix3D(
      ThreeD(outputRows, outputColumns, numbFilters),
      new Array[Double](outputColumns * outputRows * numbFilters)
//      for (_ <- (0 until outputColumns * outputRows * numbFilters).toArray) yield rng.between(-1.0, 1.0)
    )
    w.attachGrad()
//    b.attachGrad()

    Convolution2D(weights = w, biases = b, stride, activation)
  }
}