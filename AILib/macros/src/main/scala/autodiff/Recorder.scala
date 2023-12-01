package autodiff

import tensors.*
import tensors.TensorWrapper.*
import utils.TensorOperators.*
import utils.Counter

import scala.quoted.*

/**
 * calls recordImpl with an expression of the function and an expression of the product
 *
 * @param f function of type T1 => T2
 * @param prod product the function 'f' lives in
 * @tparam T1 parameter type as subtype of 'Tensor'
 * @tparam T2 output type as subtype of 'Tensor'
 * @return a function of the same input output mapping as 'f'
 */
inline def record[T1 <: Tensor, T2 <: Tensor](inline f: T1 => T2, prod: Product): T1 => T2 = ${
  Recorder.recordImpl('f, 'prod)
}

/**
 * calls recordImpl with an expression of the function and an expression of the product
 *
 * @param f function of type (T1, T2) => T3
 * @param prod pruduct the function 'f' lives in
 * @tparam T1 first parameter type as subtype of 'Tensor'
 * @tparam T2 second parameter type as subtype of 'Tensor'
 * @tparam T3 output type as subtype of 'Tensor'
 * @return a functino of the same input output mapping as 'f'
 */
inline def record[T1 <: Tensor, T2 <: Tensor, T3 <: Tensor](inline f: (T1, T2) => T3, prod: Product): (T1, T2) => T3 = ${
  Recorder.recordImpl('f, 'prod)
}

/**
 * Recorder object containing necessary values and functions for the recordImpl functions
 */
object Recorder {
  // mapping of the functionID(product.hashCode) to a graph
  // avoids evaluating the graphs obtained by decoding the function on each function call
  var IDGraphMap: Map[Int, Graph] = Map()

  // mapping of the functionID to the parameterIDs
  var pIDsMap: Map[Int, List[Int]] = Map()

  /**
   * assigns the computational graph that created 'res' to it given the product 'prod'
   *
   * @param fID identifyer of the function that results in 'res'
   * @param res the 'Tensor' obtaining the computational graph
   * @tparam T type of 'res' as subtype of 'Tensor'
   */
  def graphWithID[T <: Tensor](fID: Int, res: T): Unit = {
    res.setGraph(IDGraphMap.get(fID).get)
  }

  /**
   * sets the ParameterLeaf's value to the value 'value'; the 'ParameterLeaf' is found in the
   * Decoder's 'parameterLeaves' map from the product's hashCode and the parameterID 'pID'
   *
   * @param fID identifyer of the function that the 'value' comes from
   * @param pID parameter ID
   * @param value tensor value that the function with 'fID' is called with
   */
  def setParameterValue(fID: Int, pID: Int, value: Tensor): Unit = {
    Decoder
      .parameterLeaves
      .get(fID)
      .get(pID)
      .setValue(value)
  }

  /**
   * adds the computational graph of 'f' to the output 'Tensor'
   * and assigns the parameter leaves in the graph values once the 'f' is called
   * while keeping the input output mapping of the 'f'
   *
   * @param fExpr expression of a function of type T1 => T2
   * @param prodExpr expression of the product the 'f' lives in
   * @tparam T1 parameter type as subtype of 'Tensor'
   * @tparam T2 output type as subtype of 'Tensor'
   * @return the function 'f' with additional functionality under the hood
   */
  def recordImpl[T1 <: Tensor, T2 <: Tensor](fExpr: Expr[T1 => T2], prodExpr: Expr[Product])
                                            (using Type[T1], Type[T2], Quotes): Expr[T1 => T2] = {
    import quotes.reflect.*


    val t: (Expr[Graph], Expr[List[Int]]) = Decoder.decode(fExpr, prodExpr)
    val graphExpr: Expr[Graph] = t._1
    val parameterIDs: Expr[List[Int]] = t._2


    '{
      (x: T1) => {
        val functionID = $prodExpr.hashCode
        lazy val graph = $graphExpr           // only evaluate when function is called the first time
        val res = $fExpr(x)                   // compute the result of the function
        // set the graph of res to the graph found in combination to the function ID
        // from the IDGraphMap, or create a new mapping
        res.setGraph(
          IDGraphMap.getOrElse(functionID, {
              IDGraphMap += functionID -> graph
              graph
            })
        )
        // get the parameter IDs from the 'pIDsMap', or create a mapping
        val pIDs = pIDsMap.getOrElse(functionID, {
          val pIDs = $parameterIDs
          pIDsMap += $prodExpr.hashCode -> pIDs
          pIDs
        })
        // set the parameters the values this function is called with
        Recorder.setParameterValue(functionID, pIDsMap.get(functionID).get.head, x)
        res
      }
    }
  }

  /**
   * adds the computational graph of 'f' to the output 'Tensor'
   * and assigns the parameter leaves in the graph values once the 'f' is called
   * while keeping the input output mapping of the 'f'
   *
   * @param fExpr expression of a function of type T1 => T2
   * @param prodExpr expression of the product the 'f' lives in
   * @tparam T1 first parameter type as subtype of 'Tensor'
   * @tparam T2 second parameter type as subtype of 'Tensor'
   * @tparam T3 output type as subtype of 'Tensor'
   * @return the function 'f' with additional functionality under the hood
   */
  def recordImpl[T1 <: Tensor, T2 <: Tensor, T3 <: Tensor]
  (fExpr: Expr[(T1, T2) => T3], prodExpr: Expr[Product])
  (using Type[T1], Type[T2], Type[T3], Quotes): Expr[(T1, T2) => T3] = {

    import quotes.reflect.*
    val t: (Expr[Graph], Expr[List[Int]]) = Decoder.decode(fExpr, prodExpr)
    val graphExpr: Expr[Graph] = t._1
    val parameterIDs: Expr[List[Int]] = t._2

    '{
      (x1: T1, x2: T2) => {
        val functionID = $prodExpr.hashCode
        lazy val graph = $graphExpr                // only evaluate when function is called the first time
        val res = $fExpr(x1, x2)                   // compute the result of the function
        // set the graph of res to the graph found in combination to the function ID
        // from the IDGraphMap, or create a new mapping
        res.setGraph(IDGraphMap.getOrElse(functionID, {
          IDGraphMap += functionID -> graph
          graph
        }))
        // get the parameter IDs from the 'pIDsMap', or create a mapping
        val pIDs = pIDsMap.getOrElse(functionID, {
          val pIDs = $parameterIDs
          pIDsMap += $prodExpr.hashCode -> pIDs
          pIDs
        })
        // set the parameters the values this function is called with
        Recorder.setParameterValue(functionID, pIDsMap.get(functionID).get(1), x1)
        Recorder.setParameterValue(functionID, pIDsMap.get(functionID).get.head, x2)
        res
      }
    }
  }
}