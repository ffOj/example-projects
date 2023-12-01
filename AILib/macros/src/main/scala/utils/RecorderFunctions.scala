package utils

import tensors.*
import tensors.Tensor.*
import TensorOperators.*

object RecorderFunctions {
  def plus(t1: Tensor, t2: Tensor): Tensor = t1 + t2
  def minus(t1: Tensor, t2: Tensor): Tensor = t1 - t2
  def times(t1: Tensor, t2: Tensor): Tensor = t1 * t2
  def divides(t1: Tensor, t2: Tensor): Tensor = t1 / t2

  def plus(t: Tensor, d: Double): Tensor = t + d
  def minus(t: Tensor, d: Double): Tensor = t - d
  def times(t: Tensor, d: Double): Tensor = t * d
  def divides(t: Tensor, d: Double): Tensor = t / d

  def plus(d: Double, t: Tensor): Tensor = d + t
  def minus(d: Double, t: Tensor): Tensor = d - t
  def times(d: Double, t: Tensor): Tensor = d * t
  def divides(d: Double, t: Tensor): Tensor = d / t

  def plus(d1: Double, d2: Double): Double = d1 + d2
  def minus(d1: Double, d2: Double): Double = d1 - d2
  def times(d1: Double, d2: Double): Double = d1 * d2
  def divides(d1: Double, d2: Double): Double = d1 / d2

  def dot(t1: Tensor, t2: Tensor): Tensor = t1 `@` t2
  def T(t: Tensor): Tensor = t.T
  def square(t: Tensor): Tensor = t.map(math.pow(_, 2))
  
  def log(t: Tensor): Tensor = t.map(math.log)
  def sum(t: Tensor): Tensor = Scalar(t.sum)

  def convolution2D(weights: Tensor,
                    input: Tensor,
                    stride: (Int, Int)): Tensor = (weights, input) match {
    case (weights: Matrix4D, input: Matrix3D) => convolution2D_(weights, input, stride)
    case (Matrix3D(ThreeD(r,c,d), elements), input: Matrix3D) => {
      val nWeights = Matrix4D(FourD(r,c,d,1), elements)
      convolution2D_(nWeights, input, stride)
    }
    case _ => ???
  }

  private def convolution2D_(weights: Matrix4D, input: Matrix3D, stride: (Int, Int)): Matrix3D = {
    val kernels = weights.toMatrix3Ds

    val outputRows: Int = (input.shape.rows - weights.shape.rows) / stride._1 + 1
    val outputCols: Int = (input.shape.columns - weights.shape.columns) / stride._2 + 1
    val outputDepth: Int = kernels.length

    val resArray = new Array[Double](outputRows * outputCols * outputDepth)
    val res = Matrix3D(ThreeD(outputRows, outputCols, outputDepth), resArray)

    for (r <- 0 until outputRows) {
      for (c <- 0 until outputCols) {
        for ((kernel, d) <- kernels.zipWithIndex) {
          val idx = res.indexOf(r, c, d)
          for (kr <- 0 until kernel.shape.rows) {
            for (kc <- 0 until kernel.shape.columns) {
              for (kd <- 0 until kernel.shape.depth) {
                val k = kernel.elementWithIndices(kr, kc, kd)
                val i = input.elementWithIndices(r*stride._1 + kr, c*stride._2 + kc, kd)
                res.elements(idx) += k * i
              }
            }
          }
        }
      }
    }
    res
  }

  // TODO: stride is not implemented yet!!!
  def flippedFullConvolution2D(weights: Tensor, input: Tensor, stride: (Int, Int)): Tensor = (weights, input) match {
    case (weights: Matrix4D, input: Matrix3D) => flippedFullConvolution2D_(weights, input, stride)
    case _ => ???
  }

  def flippedFullConvolution2D_(weights: Matrix4D, input: Matrix3D, stride: (Int, Int)): Matrix3D = {
    val kernels: List[Matrix3D] = weights.toMatrix3Ds.map(_.flip180)

    val outputRows: Int = input.shape.rows + weights.shape.rows + (stride._1*2) - 3 // -2 due to overlap -1 due to stride
    val outputCols: Int = input.shape.columns + weights.shape.columns + (stride._2*2) - 3
    val outputDepth: Int = input.shape.depth

    val outputArray = new Array[Double](outputRows * outputCols * outputDepth)
    val output = Matrix3D(ThreeD(outputRows, outputCols, outputDepth), outputArray)

    for (totalRows <- 0 until input.shape.rows + weights.shape.rows) {
      for (totalCols <- 0 until input.shape.columns + weights.shape.columns) {
        for ((kernel, d) <- kernels.zipWithIndex) {
          for (kr <- 0 until kernel.shape.rows) {
            for (kc <- 0 until kernel.shape.columns) {
              for (kd <- 0 until kernel.shape.depth) {
                val idx = output.indexOf(totalRows, totalCols, d)
                val inputRow: Int = totalRows - ((outputRows - input.shape.rows) / 2) + kr - 1
                val inputCol: Int = totalCols - ((outputCols - input.shape.columns) / 2) + kc - 1
                if (0 <= inputRow && inputRow < input.shape.rows && 0 <= inputCol && inputCol < input.shape.columns) {
                  output.elements(idx) +=
                    kernel.elementWithIndices(kr, kc, kd) * input.elementWithIndices(inputRow, inputCol, kd)
                }
              }
            }
          }
        }
      }
    }
    output
  }

  def flatten(t: Tensor): Tensor = CVector(t.dimension.asList.product, t.elements)

  def maxpool2D(t: Tensor, windowSize: (Int, Int), stride: (Int, Int)): Tensor = {
    val input = t.asInstanceOf[Matrix3D]

    val outputRows: Int = (input.shape.rows - windowSize._1) / stride._1 + 1
    val outputCols: Int = (input.shape.columns - windowSize._2) / stride._2 + 1
    val outputDepth: Int = input.shape.depth

    val outputArray = new Array[Double](outputRows * outputCols * outputDepth)
    val output = Matrix3D(ThreeD(outputRows, outputCols, outputDepth), outputArray)

    for (r <- 0 until outputRows) {
      for (c <- 0 until outputCols) {
        for (d <- 0 until outputDepth) {
          var max: Option[Double] = None
          for (kr <- 0 until windowSize._1) {
            for (kc <- 0 until windowSize._2) {
              val i = input.elementWithIndices(r*stride._1 + kr, c*stride._2 + kc, d)
              if (max.isEmpty || i > max.get) max = Option(i)
            }
          }
          output.elements(output.indexOf(r, c, d)) = max.get
        }
      }
    }
    output
  }
  
  def argmax(t: Tensor, windowSize: (Int, Int), stride: (Int, Int)): Tensor = {
    val input = t.asInstanceOf[Matrix3D]

    val outputRows: Int = ((input.shape.rows - windowSize._1) / stride._1) + 1
    val outputCols: Int = ((input.shape.columns - windowSize._2) / stride._2) + 1
    val outputDepth: Int = input.shape.depth

    val outputArray = new Array[Double](outputRows * outputCols * outputDepth)
    val output = Matrix3D(ThreeD(outputRows, outputCols, outputDepth), outputArray)

    for (r <- 0 until outputRows) {
      for (c <- 0 until outputCols) {
        for (d <- 0 until outputDepth) {
          var max: Option[Double] = None
          var index = -1
          for (kr <- 0 until windowSize._1) {
            for (kc <- 0 until windowSize._2) {
              val idx = input.indexOf(r*stride._1 + kr, c*stride._2 + kc, d)
              val i = input.elements(idx)
              if (max.isEmpty || i > max.get) {
                max = Option(i)
                index = idx
              } 
            }
          }
          output.elements(output.indexOf(r, c, d)) = index
        }
      }
    }
    output
  }

  def avgpool2D(t: Tensor, windowSize: (Int, Int), stride: (Int, Int)): Tensor = {
    val input = t.asInstanceOf[Matrix3D]

    val outputRows: Int = (input.shape.rows - windowSize._1) / stride._1 + 1
    val outputCols: Int = (input.shape.columns - windowSize._2) / stride._2 + 1
    val outputDepth: Int = input.shape.depth

    val outputArray = new Array[Double](outputRows * outputCols * outputDepth)
    val output = Matrix3D(ThreeD(outputRows, outputCols, outputDepth), outputArray)

    val kernelWindow = windowSize._1 * windowSize._2
    for (r <- 0 until outputRows) {
      for (c <- 0 until outputCols) {
        for (d <- 0 until outputDepth) {
          var accumulation = 0.0
          for (kr <- 0 until windowSize._1) {
            for (kc <- 0 until windowSize._2) {
              val i = input.elementWithIndices(r*stride._1 + kr, c*stride._2 + kc, d)
              accumulation += i
            }
          }
          output.elements(output.indexOf(r, c, d)) = accumulation / kernelWindow
        }
      }
    }
    output
  }
}
