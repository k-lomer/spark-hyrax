# spark-hyrax
Distributed methods for block A-orthonormalization

# A-orthonormalization
The `lb.edu.aub.hyrax` package contains two implementations of the Classical Gram–Schmidt A-orthonormalization routines for distributed Spark clusters. The algorithms are based on the paper _Enlarged Krylov Subspace Methods and Preconditioners for Avoiding Communication_ by Sophie Moufawad. This paper presents a number of routines for Enlarged Krylov Supspace Conjugate Gradient methods, with adaptations to reduce the communication costs of performing parallel or distributed computations. The A-orthonormalization subroutine is one of the core operations of these methods and in experimental testing it was responsible for around half of the work done by the full CG method.

This implementation uses custom matrix classes for distributed operations. Please see the Matrix documentation notebook for more information on the Matrix operations

### Operation Definition
Vectors \\(u,v\in \mathbb{R}^n\\) said to be _A-orthonormal_ for a given $n\times n$ matrix \\(A\\), if \\(u^tAv=0\\) and \\(u^tAu=v^tAv=1\\)

The Block Classical Gram–Schmidt A-orthonormalization operation (BCGS) takes a \\(n\times tk\\) matrix \\(Q\\) whose columns are A-orthonormal and a \\(n\times t\\) matrix \\(P\\) whose columns will be A-orthonormalized. The return value is a \\(n\times t\\) matrix \\(P'\\) whose columns are A-orthonormal to those of \\(Q\\). For greater numerical accuracy, we apply the routine twice, first to \\((Q,P)\\) and then to \\((Q,P')\\). This is known as BCGS2.

##Regular implementation
The naive implementation of BCGS requires the sparse matrix \\(A\\) as an input and is twice multiplied by the tall skinny dense matrix \\(P\\) during the routine.  
This may involve data shuffling between partitions if there are values in \\(A\\) which are off of the block diagonal.

**Algorithm**  A-orthonormalization against previous vectors with BCGS
___
> **Input:** \\(A\\), the \\(n\times n\\) symmetrix positive definite matrix  
> **Input:** \\(Q\\), the \\(tk\\) column vectors to A-orthonormalize against  
> **Input:** \\(P\\), the \\(t\\) column vectors to be A-orthonormalized  
> **Output:** \\(\tilde{P}\\), \\(P\\) A-orthonormalized against \\(Q\\)

1. \\(W=AP\\)
2. \\(\tilde{P}= P - Q(Q^tW)\\)
3. \\(\tilde{W} = A\tilde{P}\\)
4. **for** \\(i=1:t\\) **do**
5. \\(\quad \tilde{P}(:,i)=\frac{\tilde{P}(:i)}{\sqrt{\tilde{P}(:i)^t\tilde{W}(:,i)}}\\)
6. **end for**

`aorthoBCGS2(Q: DistributedDenseMatrix, WIn: DistributedDenseMatrix, A: DistributedSparseMatrix, CGS2: Boolean, cache: Boolean=true)`

> `Q`: `DistributedDenseMatrix` of dimension _(n, tk)_, the previous _tk_ column vectors to A-orthonormalize against  
> `PIn`: `DistributedDenseMatrix` of dimension _(n, t)_, the _t_ column vectors to A-orthonormalize   
> `A`: `DistributedSpareMatrix` of dimension _(n, n)_, symmetric positive definite sparse matrix  
> `CGS2`: `Boolean`, whether to apply the CGS routine twice  
> `cache`: `Boolean`, whether to cache intermediate results which are used more than once (recommended `true`)  

> output: `DistributedDenseMatrix` of dimension _(n, t)_, the column vectors of `PIn` A-orthonormalized against `Q`

#### Example Usage
See the Load data documentation for more information on the data preprocessing

```Scala
import lb.edu.aub.hyrax.Aortho.aorthoBCGS
import lb.edu.aub.hyrax.DistributedDenseMatrix
import lb.edu.aub.hyrax.DistributedSparseMatrix
import lb.edu.aub.hyrax.FixedRangePartitioner
import lb.edu.aub.hyrax.LoadData.{readDenseMatRDD, readSparseMatRDD}

val dir = "test_data/"

// size of the matrix
val n = 10000
// define partitions
val partitions: Array[(Long, Long)] = Array((0, 2499), (2500, 4999), (5000, 7499), (7500, 9999))
// create partitioner
val p = new FixedRangePartitioner(partitions)

// Load data
val Q = new DistributedDenseMatrix(readDenseMatRDD(dir + "Q.out", sc).partitionBy(p).cache)
val P = new DistributedDenseMatrix(readDenseMatRDD(dir + "P.out", sc).partitionBy(p).cache)
val A = new DistributedSparseMatrix(readSparseMatRDD(dir + "A.out", sc, false, false).partitionBy(p).cache, n)

val newP = aorthoBCGS(Q, P, A, true)

println("Check values are Aortho")
println("Against Q")
println(Q.dot(A * newP)) // Check columns of P are Aorthogonal to Q (Q^t * A * P = 0)

println("Check values against themselves are scaled to 1")
println(newP.diagonalDot(A * newP)) // Check columns of P are Aorthonormal (P_i^t * A * P_i = 1)
```
