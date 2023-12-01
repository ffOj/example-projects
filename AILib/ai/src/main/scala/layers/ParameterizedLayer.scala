package layers

import tensors.Tensor
import tensors.TensorWrapper.*

trait ParameterizedLayer[W <: Tensor, B <: Tensor] extends Layer {
  var weights: W
  var biases: B

  def update(learningRate: Double): Unit = {
    import tensors.TensorWrapper.*
    import utils.TensorOperators.*

    val wGrad = weights.getGrad.getGradient
    val bGrad = biases.getGrad.getGradient

    weights.elements = (weights - (learningRate * wGrad)).elements
    biases.elements = (biases - (learningRate * bGrad)).elements
  }

  def accumulateGradients(): Unit = {
    if (weights.hasGrad) weights.getGrad.accumulate()
    if (biases.hasGrad) biases.getGrad.accumulate()
  }

  def stripAccumulation(): Unit = {
    if (weights.hasGrad) weights.getGrad.stripAccumulation()
    if (biases.hasGrad) biases.getGrad.stripAccumulation()
  }
}