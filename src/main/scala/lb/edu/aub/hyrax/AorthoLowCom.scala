package lb.edu.aub.hyrax

import breeze.numerics.sqrt

object AorthoLowCom {
  /*The function A-orthonormalizes the given vectors W against the vectors
  of Q using the classical Gram Schmidt procedure with reorthonormalization.

  // input Q: DistributedDenseMatrix, dense matrix whose columns are previous basis vectors to A-orthonormalize against
  // input AQ: DistributedDenseMatrix, dense matrix A*Q
  // input W: DistributedDenseMatrix, a dense matrix whose columns are the set of vectors to A-orthormalize
  // input AW: DistributedDenseMatrix, dense matrix A*W
  // input CGS2: Boolean, whether to use the CGS2 re-orthonormalization

  // output W: DistributedDenseMatrix, dense matrix W A-ortonormalized wrt to the columns of Q
  // output AW: DistributedDenseMatrix, A*W
  */
  def aorthoBCGSLowCom( Q: DistributedDenseMatrix,  AQ: DistributedDenseMatrix,
                          WIn: DistributedDenseMatrix,  AWIn: DistributedDenseMatrix,
                          CGS2: Boolean, cache: Boolean = true):
  (DistributedDenseMatrix, DistributedDenseMatrix) = {
    val proj = Q.dot(AWIn)
    val newW = WIn - (Q * proj)
    val newAW = AWIn - (AQ * proj)

    if (cache) {
      newW.cache
      newAW.cache
    }

    // scale columns of W to magnitude 1
    val ZZ = sqrt(newW.diagonalDot(newAW)).toArray
    if (ZZ contains 0.0) throw new ArithmeticException("Cannot divide by zero")
    val scaledNewW = newW / ZZ
    val scaledNewAW = newAW / ZZ


    if(CGS2) {
      val ret = aorthoBCGSLowCom(Q, AQ, scaledNewW, scaledNewAW, false, cache)
      (ret._1, ret._2)
    }
      else (scaledNewW, scaledNewAW)
  }

}
