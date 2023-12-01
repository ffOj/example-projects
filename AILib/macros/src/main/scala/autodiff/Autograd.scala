package autodiff

import tensors.Tensor
import utils.TensorOperators.*
import autodiff.Graph

/**
 * Stores gradients
 */
class Autograd {
  var gradient: Option[Tensor] = None

  private var accumulatedGradient: Option[Tensor] = None

  /**
   * sets accumulatedGradient to None
   */
  def stripAccumulation(): Unit = accumulatedGradient = None

  /**
   * accumulates gradient to current accumulatedGradient,
   * if accumulatedGradient is not defined, it defines it to the new gradient
   */
  def accumulate(): Unit = {
    if (accumulatedGradient.isDefined) {
      accumulatedGradient = Option((accumulatedGradient.get + gradient.get) / 2)
    } else { accumulatedGradient = Option(gradient.get) }
    gradient = None
  }

  /**
   * gets the accumulated gradient if it exists and strips it afterwards,
   * otherwise it gets the result of the gradient graph
   *
   * @return accumulatedGradient if accumulatedGradient is defined, gradient.compute otherwise
   */
  def getGradient: Tensor = {
    val res = accumulatedGradient.getOrElse({
      gradient.get
    })
    gradient = None
    accumulatedGradient = None
    res
  }

  /**
   * setter for the gradient
   *
   * @param t new value of the gradient
   */
  def addGradient(t: Tensor): Unit = {
    if (gradient.isDefined) {
      gradient = Option(gradient.get + t)
    } else gradient = Option(t)
  }

  /**
   * check whether a gradient is defined
   *
   * @return true if gradient is defined, false otherwise
   */
  def hasGradient: Boolean = gradient.isDefined
}