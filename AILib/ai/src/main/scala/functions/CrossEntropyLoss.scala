package functions

import tensors.Tensor
import utils.RecorderFunctions.*

import autodiff.record

case class CrossEntropyLoss() extends CostFunction {
  override lazy val calculate: (Tensor, Tensor) => Tensor = record(
    (y, yHat) => minus(
      0,
      sum(
        plus(
          times(
            y,
            log(yHat)
          ),
          times(
            minus(1, y),
            log(
              minus(1, yHat)
            )
          )
        )
      )
    ), this
  )

}
