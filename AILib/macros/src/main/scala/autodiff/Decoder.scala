package autodiff

import tensors.Tensor
import utils.Counter

import scala.quoted.*

import scala.collection.mutable.Stack

/**
Object to decode a function that lives in a Product for easy reference to the product's elements
 */
object Decoder {
  // product hashCode -> (parameterID -> ParameterLeaf)
  var parameterLeaves: Map[Int, Map[Int, ParameterLeaf]] = Map()

  val parameterID: Counter = Counter()

  /**
   * decodes the function expression into a computational graph
   * Note: works only for single line functions using postfix notation
   *
   * @param fExpr function expression
   * @param prodExpr product expression
   * @param _ function type
   * @return tuple of computational graph expression and
   *         list expression of the identifying integers for the parameter leaves -
   *         those can be found in the 'parameterLeaves' map using the product's
   *         hashCode and the elements of this list; the IDs have the same order as the
   *         parameters of the function
   */
  def decode(fExpr: Expr[_], prodExpr: Expr[Product])
            (using Quotes): (Expr[Graph], Expr[List[Int]]) = {
    import quotes.reflect.*

    var operatorNames: Stack[String] = Stack()

    // parameter names
    var parameters: List[String] = List()

    /**
     * compares the name to the supported operator names as well as the string "activation"
     * and returns true if it is in the supported operators and not "activation"
     *
     * @param name possible operator name
     * @return true if name is in supported operators found in the 'Operator' object and not "activation";
     *         false otherwise
     */
    def isFunctionName(name: String): Boolean = {
      Operator.supportedOps.contains(name) && name != "activation"
    }

    /**
     * decodes a syntax tree into a computational graph expression
     *
     * @param tree syntax tree
     * @return expression of the computational graph representing the tree, 'Empty' if none can be created
     */
    def decode_(tree: Tree): Expr[Graph] = {
      tree match {
        case Inlined(_, _, t) => decode_(t)
        // for multiline code, this will be a list of all lines
        case Block(List(t), _) => decode_(t)
        case DefDef(_, List(l), _, optT) => {
          for (t <- l.asInstanceOf[List[Tree]]) decode_(t)
          decode_(optT.get)
        }
        // contains the parameter names
        case ValDef(name, _, _) => {
          parameters = name :: parameters
          '{ Empty }
        }
        // t is the function name and l a list of parameters
        case Apply(t, l) => {
          decode_(t)
          // decodes each element in the list l and removes 'Empty's to obtain the function parameters
          val argList = l.map(e => decode_(e)).filter(_ != Empty)
          Graph(
            argList,
            // operator names are taken from a stack,
            // in case of "activation", the first name is the name of the ActivationFunction object
            //    and the second is the function call "activate"
            // in all other cases, the first element is "", the second is the operator name
            s"${operatorNames.pop()}.${operatorNames.pop()}",
            prodExpr
          )
        }
        case TypeApply(t, _) => decode_(t)
        // contains the operator name in 'operator' and the object of the location of the operator function in t
        // this is different from other 'Ident's, as they usually signal function parameters
        case Select(t, operator) => {
          operatorNames.push(operator)
          t match {
            case Ident(name) => operatorNames.push(name)
            case This(name) => operatorNames.push(name.toString)
            case _ => operatorNames.push("")
          }
          '{ Empty }
        }
        // literals are transformed to ConstantDouble nodes with the value in 'const'
        case Literal(const) => '{ ConstantDouble(${ Expr(const.value.asInstanceOf[Double]) }) }
        case Ident(name) => {
          // if the name is a parameter name of the function in 'fExpr', a 'ParameterLeaf' is returned, additionally,
          // if a 'ParameterLeaf' of this name already exists in the 'parameterLeaves' map, that is returned,
          // otherwise it is added to the map
          if (parameters.contains(name)) '{
            // get the product's parameterID -> ParameterLeaf map, or else create it
            parameterLeaves.getOrElse($prodExpr.hashCode, {
              val p = ParameterLeaf(${Expr(name)})
              parameterLeaves += $prodExpr.hashCode -> Map(parameterID.assign -> p)
              Map(parameterID.current -> p)
            })
              // find the map whith name being in one of the ParameterLeafs' names,
              // or else add the new map to the product's hash code map
              .find(t => t._2.name == ${Expr(name)}).getOrElse({
              val p = ParameterLeaf(${ Expr(name) })
              val map = parameterLeaves.get($prodExpr.hashCode).get
              parameterLeaves += $prodExpr.hashCode -> (map + (parameterID.assign -> p))
              -1 -> p
            })._2
          // if the name is a function name, add that name to the 'operatorNames' stack
          } else if (isFunctionName(name)) {
            operatorNames.push(name)
            operatorNames.push("")
            '{ Empty }
          // else, return a 'Leaf' of the name's index in the product
          } else '{
            Leaf(
              $prodExpr
                .productElementNames
                .indexOf(${ Expr(name) }),
              $prodExpr
            )
          }
        }
        case This(name) => {
          operatorNames.push(name.toString)
          '{ Empty }
        }
        case _ => '{ Empty }
      }
    }

    (
      decode_(fExpr.asTerm),
      '{ parameterID.collect() }
    )

  }

}