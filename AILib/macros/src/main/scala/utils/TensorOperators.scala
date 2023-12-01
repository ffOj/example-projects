package utils

import tensors.Tensor

object TensorOperators {
  extension (t: Tensor) {
    def +(t2: Tensor): Tensor = t.plus(t2)
    def -(t2: Tensor): Tensor = t.minus(t2)
    def *(t2: Tensor): Tensor = t.times(t2)
    def /(t2: Tensor): Tensor = t.divides(t2)

    def +(d: Double): Tensor = t.plus(d)
    def -(d: Double): Tensor = t.minus(d)
    def *(d: Double): Tensor = t.times(d)
    def /(d: Double): Tensor = t.divides(d)
  }

  extension (t: Tensor) {
    def `@`(t2: Tensor): Tensor = t.dot(t2)
    def T: Tensor = t.transpose
  }

  extension (d: Double) {
    def +(t: Tensor): Tensor = t.map(d + _)
    def -(t: Tensor): Tensor = t.map(d - _)
    def *(t: Tensor): Tensor = t.map(d * _)
    def /(t: Tensor): Tensor = t.map(d / _)
  }
}
