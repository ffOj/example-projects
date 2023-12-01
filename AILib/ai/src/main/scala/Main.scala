
import layers.{AvgPooling, Convolution2D, Dense, Flatten, MaxPooling2D}
import networks.FeedForward
import utils.RecorderFunctions.*
import utils.TensorOperators.*
import tensors.TensorWrapper.*
import autodiff.*
import functions.*
import tensors.*
import tensors.Tensor.*
import utils.ActivationFunction

import scala.math.{abs, max}

object Main extends App {
//  derivativeComps()

  def derivativeComps(): Unit = {
    val layer = Dense(5,3, Sigmoid)
    val layer2 = Dense(3,2, Sigmoid)
    val input = CVector(5, 1,2,3,4,5)

    val res = layer2.predict(layer.predict(input))
    res.backwards()

    println("actual gradients")
    println(layer.weights.getGrad.gradient.get.elements.toList)
    println(layer.biases.getGrad.gradient.get.elements.toList)
    println(layer2.weights.getGrad.gradient.get.elements.toList)
    println(layer2.biases.getGrad.gradient.get.elements.toList)
    println()
    println("correct gradients")
    val w = layer.weights
    val b = layer.biases
    val w2 = layer2.weights
    val b2 = layer2.biases

    println((w2.T `@` Sigmoid.derive(w2`@`Sigmoid.activate(w `@` input + b) + b2) * (Sigmoid.derive(w `@` input + b) `@` input.T)).elements.toList)
    println((w2.T `@` Sigmoid.derive(w2`@`Sigmoid.activate(w `@` input + b) + b2) * Sigmoid.derive(w `@` input + b)).elements.toList)

    println((Sigmoid.derive(w2`@`Sigmoid.activate(w `@` input + b) + b2) `@` Sigmoid.activate(w `@` input + b).T).elements.toList)
    println(Sigmoid.derive(w2`@`Sigmoid.activate(w `@` input + b) + b2).elements.toList)
  }

  // training goes much slower when increasing the number of kernels
  // second conv layer breaks it
  // same problem with avgpool and max pool that its learing after a certain time and after a short
  // time learing, it goes back to 1.8 and eventually NaN
  def MNISTclassification(): Unit = {
    val layer1 = Convolution2D(5,5,1,5,24,24, (1,1), ReLu) // reduction to 24x24x3
    val layer2 = MaxPooling2D((2,2), (2,2), 1) //  reduction to 12x12x3
    val layer3 = Convolution2D(3,3,5,8,10,10, (1,1), ReLu) // 10x10x3
    val layer4 = MaxPooling2D((2,2), (2,2), 2) // reduction to 5x5x3
    val layer5 = Flatten(fromSize = List(5,5,8))
    val layer6 = Dense(5*5*8, 120, ReLu)
    val layer7 = Dense(120, 84, ReLu)
    val layer8 = Dense(84, 10, Softmax)

    val layers = List(layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8)
    val net = FeedForward(layers, CrossEntropyLoss())

    val trainingData = Reader.readMNIST("mnist_train.csv")
    val testData = Reader.readMNIST("mnist_test.csv")

    for ((in, actual) <- testData.slice(0, 10)) {
      val pred = net.predict(in)
      println((actual.elements.zipWithIndex.foldLeft((-1.0, -1))((z, i) => if (z._1 > i._1) z else i)._2, pred.elements.toList))
    }

    val iterations = 100
    val batchSize = 1
    val lr = 0.01

    net.stochasticGradientDescent(testData.slice(0, 10), batchSize, iterations, lr)

    for ((in, actual) <- testData.slice(0, 10)) {
      val pred = net.predict(in)
      println((actual.elements.zipWithIndex.foldLeft((-1.0, -1))((z, i) => if (z._1 > i._1) z else i)._2, pred.elements.toList))
    }
    val finalError = net.evaluate(testData.slice(0, 10), error)
    println(s"generalization error after training: $finalError")
  }

  MNISTclassification()

  def error(prediction: Tensor, actual: Tensor): Double =
    (actual - prediction).elements.foldLeft(0.0)(_ + abs(_))
}