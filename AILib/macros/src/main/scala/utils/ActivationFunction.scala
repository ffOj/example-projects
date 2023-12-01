package utils

import tensors.Tensor
import autodiff.Graph

trait ActivationFunction {
  def activate(t: Tensor): Tensor
  
  def derive(t: Tensor): Tensor
}
