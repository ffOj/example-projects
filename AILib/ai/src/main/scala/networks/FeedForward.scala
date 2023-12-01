package networks

import functions.CostFunction
import layers.{Layer, ParameterizedLayer}
import tensors.Tensor
import tensors.TensorWrapper.*

import scala.util.Random


case class FeedForward(layers: List[Layer], costFunction: CostFunction) {
  private lazy val parameterizedLayers: List[ParameterizedLayer[_, _]] = layers
    .filter(_.isInstanceOf[ParameterizedLayer[_, _]])
    .map(_.asInstanceOf[ParameterizedLayer[_, _]])

  def predict(input: Tensor): Tensor = layers.foldLeft(input)((i, l) => l.predict(i))

  def evaluate(data: List[(Tensor, Tensor)], evaluationMetric: (Tensor, Tensor) => Double): Double = {
    def evaluate_(data: List[(Tensor, Tensor)]): Double = data match {
      case Nil => 0
      case (input, actual) :: tail => evaluationMetric(predict(input), actual) + evaluate_(tail)
    }

    evaluate_(data) / data.length
  }

  def backpropagate(input: Tensor, expectation: Tensor, learningRate: Double): Unit = {
    val output = predict(input)
    val cost = costFunction.calculate(expectation, output)

    cost.backwards()
    parameterizedLayers.foreach(_.update(learningRate))
  }

  def stochasticGradientDescent(values: List[(Tensor, Tensor)],
                                batchSize: Int,
                                epoch: Int,
                                learningRate: Double): Unit = {
    var lr = learningRate // trainer implementation necessary
    val rng: Random = Random()
    for (e <- 0 until epoch) {
      var b = 0
      lr -= 0.001

      println(s"epoch: ${e}")
      println(s"error on training set: ${evaluate(values.slice(0, 10), error)}")

      for (iter <- 0 until values.length) {
        b = (b+1) % batchSize
        val t = values(iter)//rng.between(0, values.length))
        val output = predict(t._1)
        val cost = costFunction.calculate(t._2, output)
        cost.backwards()

        parameterizedLayers.foreach(_.accumulateGradients())

        if (b == 0) parameterizedLayers.foreach(l => {
          l.update(learningRate)
          l.stripAccumulation()
        })
      }
    }
  }

  def error(prediction: Tensor, actual: Tensor): Double = {
    import utils.TensorOperators.*
    (actual - prediction).elements.foldLeft(0.0)(_ + scala.math.abs(_))
  }
}


