# spark-hyrax
Distributed methods for block A-orthonormalization

# A-orthonormalization
The `lb.edu.aub.hyrax` package contains two implementations of the Classical Gram–Schmidt A-orthonormalization routines for distributed Spark clusters. The algorithms are based on the paper _Enlarged Krylov Subspace Methods and Preconditioners for Avoiding Communication_ by Sophie Moufawad. This paper presents a number of routines for Enlarged Krylov Supspace Conjugate Gradient methods, with adaptations to reduce the communication costs of performing parallel or distributed computations. The A-orthonormalization subroutine is one of the core operations of these methods and in experimental testing it was responsible for around half of the work done by the full CG method.

This implementation uses custom matrix classes for distributed operations. Please see the Matrix documentation notebook for more information on the Matrix operations

### Operation Definition
Vectors <img src="https://render.githubusercontent.com/render/math?math=u,v\in \mathbb{R}^n"> said to be _A-orthonormal_ for a given <img src="https://render.githubusercontent.com/render/math?math=n\times n"> matrix <img src="https://render.githubusercontent.com/render/math?math=A">, if <img src="https://render.githubusercontent.com/render/math?math=u^tAv=0"> and <img src="https://render.githubusercontent.com/render/math?math=u^tAu=v^tAv=1">

The Block Classical Gram–Schmidt A-orthonormalization operation (BCGS) takes a <img src="https://render.githubusercontent.com/render/math?math=n\times tk"> matrix <img src="https://render.githubusercontent.com/render/math?math=Q"> whose columns are A-orthonormal and a <img src="https://render.githubusercontent.com/render/math?math=n\times t"> matrix <img src="https://render.githubusercontent.com/render/math?math=W"> whose columns will be A-orthonormalized. The return value is a <img src="https://render.githubusercontent.com/render/math?math=n\times t"> matrix <img src="https://render.githubusercontent.com/render/math?math=W'"> whose columns are A-orthonormal to those of <img src="https://render.githubusercontent.com/render/math?math=Q">. For greater numerical accuracy, we apply the routine twice, first to <img src="https://render.githubusercontent.com/render/math?math=(Q,W)"> and then to <img src="https://render.githubusercontent.com/render/math?math=(Q,W')">. This is known as BCGS2.

##Regular implementation
The naive implementation of BCGS requires the sparse matrix <img src="https://render.githubusercontent.com/render/math?math=A"> as an input and is twice multiplied by the tall skinny dense matrix <img src="https://render.githubusercontent.com/render/math?math=W"> during the routine.  
This may involve data shuffling between partitions if there are values in <img src="https://render.githubusercontent.com/render/math?math=A"> which are off of the block diagonal.

**Algorithm**  A-orthonormalization against previous vectors with BCGS
___
> **Input:** <img src="https://render.githubusercontent.com/render/math?math=A">, the <img src="https://render.githubusercontent.com/render/math?math=n\times n"> symmetrix positive definite matrix  
> **Input:** <img src="https://render.githubusercontent.com/render/math?math=Q">, the <img src="https://render.githubusercontent.com/render/math?math=tk"> column vectors to A-orthonormalize against  
> **Input:** <img src="https://render.githubusercontent.com/render/math?math=W">, the <img src="https://render.githubusercontent.com/render/math?math=t"> column vectors to be A-orthonormalized  
> **Output:** <img src="https://render.githubusercontent.com/render/math?math=\tilde{W}">, <img src="https://render.githubusercontent.com/render/math?math=W"> A-orthonormalized against <img src="https://render.githubusercontent.com/render/math?math=Q">

1. <img src="https://render.githubusercontent.com/render/math?math=X=AW">
2. <img src="https://render.githubusercontent.com/render/math?math=\tilde{W}= W - Q(Q^tX)">
3. <img src="https://render.githubusercontent.com/render/math?math=\tilde{X} = A\tilde{W}">
4. **for** <img src="https://render.githubusercontent.com/render/math?math=i=1:t"> **do**
5. <img src="https://render.githubusercontent.com/render/math?math=\quad \tilde{W}(:,i)=\frac{\tilde{W}(:i)}{\sqrt{\tilde{W}(:i)^t\tilde{X}(:,i)}}">
6. **end for**

`aorthoBCGS2(Q: DistributedDenseMatrix, WIn: DistributedDenseMatrix, A: DistributedSparseMatrix, CGS2: Boolean, cache: Boolean=true)`

> `Q`: `DistributedDenseMatrix` of dimension _(n, tk)_, the previous _tk_ column vectors to A-orthonormalize against  
> `WIn`: `DistributedDenseMatrix` of dimension _(n, t)_, the _t_ column vectors to A-orthonormalize   
> `A`: `DistributedSpareMatrix` of dimension _(n, n)_, symmetric positive definite sparse matrix  
> `CGS2`: `Boolean`, whether to apply the CGS routine twice  
> `cache`: `Boolean`, whether to cache intermediate results which are used more than once (recommended `true`)  

> output: `DistributedDenseMatrix` of dimension _(n, t)_, the column vectors of `WIn` A-orthonormalized against `Q`

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
val W = new FixedRangePartitioner(partitions)

// Load data
val Q = new DistributedDenseMatrix(readDenseMatRDD(dir + "Q.out", sc).partitionBy(p).cache)
val W = new DistributedDenseMatrix(readDenseMatRDD(dir + "W.out", sc).partitionBy(p).cache)
val A = new DistributedSparseMatrix(readSparseMatRDD(dir + "A.out", sc, false, false).partitionBy(p).cache, n)

println("High Communication Aortho")
val newW = aorthoBCGS(Q, W, A, true)

println("Check values are Aortho")
println("Against Q")
println(Q.dot(A * newW)) // Check columns of W are Aorthogonal to Q (Q^t * A * W = 0)

println("Check values against themselves are scaled to 1")
println(newW.diagonalDot(A * newW)) // Check columns of W are Aorthonormal (W_i^t * A * W_i = 1)
```

## Low communication implementation
To reduce the communication we do not directly multiply <img src="https://render.githubusercontent.com/render/math?math=A"> and <img src="https://render.githubusercontent.com/render/math?math=W"> but instead maintain and update a matrix containing <img src="https://render.githubusercontent.com/render/math?math=AW">.  
This removes the need to shuffle data between nodes, the only communication comes from `reduce` operations.

**Algorithm**  A-orthonormalization against previous vectors with BCGS, with low communication
___

> **Input:** <img src="https://render.githubusercontent.com/render/math?math=Q">, the <img src="https://render.githubusercontent.com/render/math?math=tk"> column vectors to A-orthonormalize against  
> **Input:** <img src="https://render.githubusercontent.com/render/math?math=W">, the <img src="https://render.githubusercontent.com/render/math?math=t"> column vectors to be A-orthonormalized  
> **Input:** <img src="https://render.githubusercontent.com/render/math?math=\mathcal{X}=AQ">  
> **Input:** <img src="https://render.githubusercontent.com/render/math?math=X=AW">  
> **Output:** <img src="https://render.githubusercontent.com/render/math?math=\tilde{W}">, <img src="https://render.githubusercontent.com/render/math?math=W"> A-orthonormalized against <img src="https://render.githubusercontent.com/render/math?math=Q">  
> **Output:** <img src="https://render.githubusercontent.com/render/math?math=\tilde{X}=A\tilde{W}">

1. <img src="https://render.githubusercontent.com/render/math?math=\tilde{W}= W - Q(Q^tX)">
2. <img src="https://render.githubusercontent.com/render/math?math=\tilde{X}= X - \mathcal{X}(Q^tX)">
4. **for** <img src="https://render.githubusercontent.com/render/math?math=i=1:t"> **do**
5. <img src="https://render.githubusercontent.com/render/math?math=\quad s_i = \sqrt{\tilde{W}(:i)^t\tilde{X}(:,i)}">
6. <img src="https://render.githubusercontent.com/render/math?math=\quad \tilde{W}(:,i)=\frac{\tilde{W}(:i)}{s_i}">
7. <img src="https://render.githubusercontent.com/render/math?math=\quad \tilde{X}(:,i)=\frac{\tilde{X}(:i)}{s_i}">
8. **end for**

`aorthoBCGS2LowCom(Q: DistributedDenseMatrix, AQ: DistributedDenseMatrix, WIn: DistributedDenseMatrix, AWIn: DistributedDenseMatrix, CGS2: Boolean, cache: Boolean=true)`

> `Q`: `DistributedDenseMatrix` of dimension _(n, tk)_, the previous _tk_ column vectors to A-orthonormalize against  
> `AQ`: `DistributedDenseMatrix` of dimension _(n, tk)_, `A * Q`  
> `WIn`: `DistributedDenseMatrix` of dimension _(n, t)_, the _t_ column vectors to A-orthonormalize   
> `AWIn`: `DistributedDenseMatrix` of dimension _(n, t)_, `A * WIn`  
> `CGS2`: `Boolean`, whether to apply the CGS routine twice  
> `cache`: `Boolean`, whether to cache intermediate results which are used more than once (recommended `true`)  

> output: (`DistributedDenseMatrix`, `DistributedDenseMatrix`) two matrices both of dimension _(n, t)_, `newW` the column vectors of `WIn` A-orthonormalized against `Q` and `A * newW`

```Scala
import lb.edu.aub.hyrax.AorthoLowCom.aorthoBCGSLowCom
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
val W = new DistributedDenseMatrix(readDenseMatRDD(dir + "W.out", sc).partitionBy(p).cache)
val A = new DistributedSparseMatrix(readSparseMatRDD(dir + "A.out", sc, false, false).partitionBy(p).cache, n)
val AQ = A * Q
val AW = A * W
AQ.cache; AW.cache

println("Low Communication Aortho")
val ret = aorthoBCGSLowCom(Q, AQ, W, AW, true) // returns (newW, newAW)

println("Check values are Aortho")
println("Against Q")
println(Q.dot(ret._2)) // Check columns of P are Aorthogonal to Q (Q^t * AW = 0)

println("Check values against themselves are scaled to 1")
println(ret._1.diagonalDot(ret._2)) // Check columns of W are Aorthonormal (W_i^t * AW_i = 1)
```
