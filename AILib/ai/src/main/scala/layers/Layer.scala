package layers

import tensors.Tensor

trait Layer {
  def predict(input: Tensor): Tensor
}
