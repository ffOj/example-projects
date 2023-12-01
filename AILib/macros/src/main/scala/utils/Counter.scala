package utils

/**
 * provides a basic counter
 * @param c starting count at and resetting to c; default -1
 * @param stepsize performs steps based on stepsize; default 1
 */
case class Counter(private val startCount: Int = -1, private val startStepsize: Int = 1) {
  private var count = startCount

  private var stepsize = startStepsize

  var assigned: List[Int] = List()

  def collect(): List[Int] = {
    val res = assigned
    assigned = Nil
    res
  }

  /**
   * takes a step according to the stepsize and returns that value
   *
   * @return the next value of the counter
   */
  def assign: Int = {
    count += stepsize
    assigned = count :: assigned
    count
  }

  /**
   * resets the count to the start count and stepsize to the starting stepsize
   */
  def reset(): Unit = {
    count = startCount
    stepsize = startStepsize
    assigned = Nil
  }

  /**
   * sets the count
   *
   * @param i new value of the counter
   */
  def setTo(i: Int): Unit = count = i

  /**
   * sets the stepsize
   *
   * @param s new value of the stepsize
   */
  def setStepsize(s: Int): Unit = stepsize = s

  /**
   * gives the current value of the counter
   *
   * @return current count
   */
  def current: Int = count
}