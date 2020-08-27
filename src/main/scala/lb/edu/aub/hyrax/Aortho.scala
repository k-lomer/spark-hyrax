package lb.edu.aub.hyrax

import breeze.numerics.sqrt

object Aortho {
  /* aorthoBCGS - Block Classical Gram-Schmidt A-orthonormalization
  The function A-orthonormalizes the given vectors W against the vectors
  of Q using the classical Gram Schmidt procedure with reorthonormalization.

  // input Q: DistributedDenseMatrix, a dense matrix whose columns are previous basis vectors to A-orthonormalize against
  // input W: DistributedDenseMatrix, a dense matrix whose columns are the set of vectors to A-orthormalize
  // input A: DistributedSparseMatrix, sparse matrix in (column, (row, value)) format
  // input CGS2: Boolean, whether to use the CGS2 re-orthonormalization

  // output W: DistributedDenseMatrix, dense  matrix W A-ortonormalized wrt to the columns of Q
  */
  def aorthoBCGS( Q: DistributedDenseMatrix,
                  WIn: DistributedDenseMatrix,  A: DistributedSparseMatrix,
                  CGS2: Boolean, cache: Boolean=true):
  DistributedDenseMatrix = {
    val AW = A * WIn
    val proj = Q.dot(AW)
    val newW = WIn - (Q * proj)
    val newAW = A * newW
    if (cache) {newW.cache}


    // scale columns of W to magnitude 1
    val ZZ = sqrt(newW.diagonalDot(newAW)).toArray
    if (ZZ contains 0.0) throw new ArithmeticException("Cannot divide by zero")
    val scaledNewW = newW / ZZ


    if(CGS2) aorthoBCGS(Q, scaledNewW, A, false, cache)
    else (scaledNewW)
  }

}
