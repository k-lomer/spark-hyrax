# spark-hyrax
Distributed methods for block A-orthonormalization

# A-orthonormalization
The `lb.edu.aub.hyrax` package contains two implementations of the Classical Gram–Schmidt A-orthonormalization routines for distributed Spark clusters. The algorithms are based on the paper _Enlarged Krylov Subspace Methods and Preconditioners for Avoiding Communication_ by Sophie Moufawad. This paper presents a number of routines for Enlarged Krylov Supspace Conjugate Gradient methods, with adaptations to reduce the communication costs of performing parallel or distributed computations. The A-orthonormalization subroutine is one of the core operations of these methods and in experimental testing it was responsible for around half of the work done by the full CG method.

This implementation uses custom matrix classes for distributed operations. Please see the Matrix documentation notebook for more information on the Matrix operations

### Operation Definition
Vectors <img src="https://render.githubusercontent.com/render/math?math=u,v\in \mathbb{R}^n"> said to be _A-orthonormal_ for a given <img src="https://render.githubusercontent.com/render/math?math=n\times n"> matrix <img src="https://render.githubusercontent.com/render/math?math=A">, if <img src="https://render.githubusercontent.com/render/math?math=u^tAv=0"> and <img src="https://render.githubusercontent.com/render/math?math=u^tAu=v^tAv=1">

The Block Classical Gram–Schmidt A-orthonormalization operation (BCGS) takes a <img src="https://render.githubusercontent.com/render/math?math=n\times tk"> matrix <img src="https://render.githubusercontent.com/render/math?math=Q"> whose columns are A-orthonormal and a <img src="https://render.githubusercontent.com/render/math?math=n\times t"> matrix <img src="https://render.githubusercontent.com/render/math?math=W"> whose columns will be A-orthonormalized. The return value is a <img src="https://render.githubusercontent.com/render/math?math=n\times t"> matrix <img src="https://render.githubusercontent.com/render/math?math=W'"> whose columns are A-orthonormal to those of <img src="https://render.githubusercontent.com/render/math?math=Q">. For greater numerical accuracy, we apply the routine twice, first to <img src="https://render.githubusercontent.com/render/math?math=(Q,W)"> and then to <img src="https://render.githubusercontent.com/render/math?math=(Q,W')">. This is known as BCGS2.

## Regular implementation
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

`aorthoBCGS(Q: DistributedDenseMatrix, WIn: DistributedDenseMatrix, A: DistributedSparseMatrix, CGS2: Boolean, cache: Boolean=true)`

> `Q`: `DistributedDenseMatrix` of dimension _(n, tk)_, the previous _tk_ column vectors to A-orthonormalize against  
> `WIn`: `DistributedDenseMatrix` of dimension _(n, t)_, the _t_ column vectors to A-orthonormalize   
> `A`: `DistributedSpareMatrix` of dimension _(n, n)_, symmetric positive definite sparse matrix  
> `CGS2`: `Boolean`, whether to apply the CGS routine twice  
> `cache`: `Boolean`, whether to cache intermediate results which are used more than once (recommended `true`)  

> output: `DistributedDenseMatrix` of dimension _(n, t)_, the column vectors of `WIn` A-orthonormalized against `Q`

#### Example Usage

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

## Communication avoiding implementation
To reduce the communication we do not directly multiply <img src="https://render.githubusercontent.com/render/math?math=A"> and <img src="https://render.githubusercontent.com/render/math?math=W"> but instead maintain and update a matrix containing <img src="https://render.githubusercontent.com/render/math?math=AW">.  
This removes the need to shuffle data between nodes, the only communication comes from `reduce` operations.

**Algorithm**  Communication avoiding A-orthonormalization against previous vectors with BCGS
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

`aorthoCABCGS(Q: DistributedDenseMatrix, AQ: DistributedDenseMatrix, WIn: DistributedDenseMatrix, AWIn: DistributedDenseMatrix, CGS2: Boolean, cache: Boolean=true)`

> `Q`: `DistributedDenseMatrix` of dimension _(n, tk)_, the previous _tk_ column vectors to A-orthonormalize against  
> `AQ`: `DistributedDenseMatrix` of dimension _(n, tk)_, `A * Q`  
> `WIn`: `DistributedDenseMatrix` of dimension _(n, t)_, the _t_ column vectors to A-orthonormalize   
> `AWIn`: `DistributedDenseMatrix` of dimension _(n, t)_, `A * WIn`  
> `CGS2`: `Boolean`, whether to apply the CGS routine twice  
> `cache`: `Boolean`, whether to cache intermediate results which are used more than once (recommended `true`)  

> output: (`DistributedDenseMatrix`, `DistributedDenseMatrix`) two matrices both of dimension _(n, t)_, `newW` the column vectors of `WIn` A-orthonormalized against `Q` and `A * newW`

```Scala
import lb.edu.aub.hyrax.AorthoCA.aorthoCABCGS
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
val ret = aorthoCABCGS(Q, AQ, W, AW, true) // returns (newW, newAW)

println("Check values are Aortho")
println("Against Q")
println(Q.dot(ret._2)) // Check columns of P are Aorthogonal to Q (Q^t * AW = 0)

println("Check values against themselves are scaled to 1")
println(ret._1.diagonalDot(ret._2)) // Check columns of W are Aorthonormal (W_i^t * AW_i = 1)
```

# Loading Data
The `lb.edu.aub.hyrax` package contains a number of helper functions and classes to help preprocessing data. Namely, the loading and partitioning of sparse and dense matrices.

## Dense Matrix
Data files are expected to be space separated values with a new line for each row.  
Values can be in decimal or scientific notation.

For example, the file for a <img src="https://render.githubusercontent.com/render/math?math=4\times 4"> dense matrix may look like:

>`-3.223e-03 -1.856e-04 -9.201e-03 8.022e-03`  
>`5.594e-03 3.538e-03 1.665e-02 -2.367e-03`  
>`-4.1 -1.709 -0.017 -6.47`  
>`0.0642 1.14 1.02 0.005`

`readDenseMatRDD(path: String, sc: SparkContext)`

> `path`: `String`, file path of the matrix  
> `sc`: `SparkContext`, the current `SparkContext`, used to parallelize the data across the cluster    

> output: `RDD[(Long, Array[Double])]` key value pairs of row indexes and row values

## Sparse Matrix
Data files are expected to be space separated triplets of row index, column index and value with a new line for each entry.  
Values can be in decimal or scientific notation.

For example, the file for a sparse matrix with 3 entries may look like:

>`1 1 0.4`  
>`2 1 -2.2`  
>`3 3 0.01`

Sometimes a symmetric sparse matrix will be stored as just the upper triangular or lower triangular entries to save space. In this case set the `reflect` parameter to `true`.

` readSparseMatrixRDD(path: String, sc: SparkContext, rowKey: Boolean=false, reflect: Boolean=false)`

> `path`: `String`, file path of the matrix  
> `sc`: `SparkContext`, the current `SparkContext`, used to parallelize the data across the cluster  
> `rowKey`: `Boolean`, whether to use the row index as the key for the output `RDD`, this should be set to `false` if the `rdd` is to be used with the `DistributedSparseMatrix` class  
> `reflect`: `Boolean`, whether to add a reflection of the off diagonal values of the matrix

> output: `RDD[(Long, (Long, Double))]` key value pairs of (row, (col, value)) if `rowKey` else (col, (row, value)) 

## Partitioning Data
Performance is best for `aorthoBCGS2` when the data is partitioned in contiguous blocks such that there are as few entries as possible in <img src="https://render.githubusercontent.com/render/math?math=A"> off the block diagonal. This reduces the amount of data that needs to be shuffled when computing <img src="https://render.githubusercontent.com/render/math?math=A\times P">. To give greater control over partitioning, we created a `FixedRangePartitioner` class. This works similarly to the `org.apache.spark.RangePartitioner` class but allows user-defined values to be chosen for the partition indexes.  For best performance, all matrices should be partitioned with the same partitioner.

`FixedRangePartitioner(val ranges: Array[(Long, Long)]) extends Partitioner`

> `ranges`: `Array[(Long, Long)]`, the start and end index for each partition

Note that the number of partitions is equal to the length of `ranges`. If the index key of an entry falls outside of the ranges given, it is placed in partition `0` by default.

```Scala
import lb.edu.aub.hyrax.DistributedDenseMatrix
import lb.edu.aub.hyrax.DistributedSparseMatrix
import lb.edu.aub.hyrax.FixedRangePartitioner
import lb.edu.aub.hyrax.LoadData.{readDenseMatRDD, readSparseMatRDD}

val dir = "/FileStore/tables/testdata/"

// size of the matrix
val n = 10000
// define partitions
val partitions:Array[(Long, Long)] = Array((0, 1999), (2000, 4999), (5000, 6999), (7000, 9999))
// create partitioner
val p = new FixedRangePartitioner(partitions)

// Load data
val Q = new DistributedDenseMatrix(readDenseMatRDD(dir + "Q.out", sc).partitionBy(p).cache)
val W = new DistributedDenseMatrix(readDenseMatRDD(dir + "W.out", sc).partitionBy(p).cache)
val A = new DistributedSparseMatrix(readSparseMatRDD(dir + "A.out", sc, false, false).partitionBy(p).cache, n)
```

# Matrix Operations
The aorthonormalization routine is composed of distributed matrix operations working on a number of matrix classes.

Three different types of matrices are used:
* `DenseMatrix` - local dense matrices from the `breeze.linalg` library
* `DistributedDenseMatrix` - our custom RDD wrapper for distributed dense matrices
* `DistributedSparseMatrix` - our custom RDD wrapper for distributed sparse matrices

Here we will review the operations provided by the custom distributed matrix classes.

## Distributed Dense Matrix


### Instance Constructors
`new DistributedDenseMatrix(rows: RDD[(Long, Array[Double])], dim: (Long, Long))`  
>`rows`: a key value pair `RDD` with values the rows in the matrix, keys the index of the row  
>`dim`: the dimensions of the matrix as (nrows, ncols)


`new DistributedDenseMatrix(rows: RDD[(Long, Array[Double])])`  
>If the `dim` parameter is not included, the constructor will use the `RDD` to look them up at a small computational overhead

note: this class makes the assumption that all rows in the matrix have an entry in the `RDD` and all value arrays have the same length.

```Scala
import lb.edu.aub.hyrax.DistributedDenseMatrix
// Create a rows RDD
val rows: RDD[(Long, Array[Double])] = sc.parallelize(Seq((0, Array(0.0, 1.0)),
                                                          (1, Array(2.0, 3.0)),
                                                          (2, Array(3.0, 4.0))))
// Create a DDM with dim
val ddm1 = new DistributedDenseMatrix(rows, (3, 2))

// Create a DDM without dim
val ddm2 = new DistributedDenseMatrix(rows)
println(ddm2.dim)
```

### Right Multiplication by a Local Matrix  
`ddm * bdm`  
> `ddm`: `DistributedDenseMatrix` of dimension _(m, n)_  
> `bdm`: `Breeze` `DenseMatrix` of dimension _(n, k)_  
>  
> output: `DistributedDenseMatrix` of dimension _(m, k)_

```Scala
import breeze.linalg.DenseMatrix

// Create a rows RDD
val rows: RDD[(Long, Array[Double])] = sc.parallelize(Seq((0, Array(0.0, 1.0)),
                                                          (1, Array(2.0, 3.0)),
                                                          (2, Array(3.0, 4.0))))
// Create a DDM
val ddm1 = new DistributedDenseMatrix(rows)
// Create a local DenseMatrix
val bdm = DenseMatrix((1.0, 2.0), (0.5, 0.5))

val ddm2 = ddm1 * bdm
```

### Matrix Addition and Subtraction  
`ddm1 + ddm2`  
`ddm1 - ddm2`
> `ddm1`: `DistributedDenseMatrix` of dimension _(m, n)_  
> `ddm2`: `DistributedDenseMatrix` of dimension _(m, n)_  
>  
> output: `DistributedDenseMatrix` of dimension _(m, n)_

```Scala
// Create rows RDDs
val rows1: RDD[(Long, Array[Double])] = sc.parallelize(Seq((0, Array(0.0, 1.0)),
                                                           (1, Array(2.0, 3.0)),
                                                           (2, Array(3.0, 4.0))))
val rows2: RDD[(Long, Array[Double])] = sc.parallelize(Seq((0, Array(1.0, 1.0)),
                                                           (1, Array(1.0, 1.0)),
                                                           (2, Array(3.0, 4.0))))

// Create DDMs
val ddm1 = new DistributedDenseMatrix(rows1)
val ddm2 = new DistributedDenseMatrix(rows2)

val ddm3 = ddm1 + ddm2

val ddm4 = ddm1 - ddm2
```

### Elementwise Division by a Single Number

`ddm / x`  

> `ddm`: `DistributedDenseMatrix` of dimension _(m, n)_  
> `x`: `Double`  
>  
> output: `DistributedDenseMatrix` of dimension _(m, n)_

```Scala
// Create rows RDD
val rows: RDD[(Long, Array[Double])] = sc.parallelize(Seq((0, Array(0.0, 1.0)),
                                                          (1, Array(2.0, 3.0)),
                                                          (2, Array(3.0, 4.0))))

// Create DDM
val ddm1 = new DistributedDenseMatrix(rows)

val ddm2 = ddm1 / 2.0
```

## Elementwise Division for Each Column
`ddm / scaleFactors`  
Divides the <img src="https://render.githubusercontent.com/render/math?math=i^{th}"> column in `ddm` by the <img src="https://render.githubusercontent.com/render/math?math=i^{th}"> entry in `v`

> `ddm`: `DistributedDenseMatrix` of dimension _(n, k)_  
> `scaleFactors`: `Array[Double]` of length _k_    
>  
> output: `DistributedDenseMatrix` of dimension _(n, k)_

```Scala
// Create rows RDD
val rows: RDD[(Long, Array[Double])] = sc.parallelize(Seq((0, Array(0.0, 1.0)),
                                                          (1, Array(2.0, 3.0)),
                                                          (2, Array(3.0, 4.0))))

// Create DDM
val ddm1 = new DistributedDenseMatrix(rows)
// Create a local DenseMatrix
val scaleFactors = Array(2.0, 0.5)

val ddm2 = ddm1 / scaleFactors
```

### Dot Product
`ddm1.dot(ddm2)`

equivalent to <img src="https://render.githubusercontent.com/render/math?math=M_1^tM_2">, dot product of the columns of the matrices

> `ddm1`: `DistributedDenseMatrix` of dimension _(n, j)_  
> `ddm2`: `DistributedDenseMatrix` of dimension _(n, k)_  
>  
> output: `Breeze` `DenseMatrix` of dimension _(j, k)_

```Scala
// Create rows RDD
val rows1: RDD[(Long, Array[Double])] = sc.parallelize(Seq((0, Array(0.0, 1.0)),
                                                           (1, Array(2.0, 3.0)),
                                                           (2, Array(3.0, 4.0))))
val rows2: RDD[(Long, Array[Double])] = sc.parallelize(Seq((0, Array(1.0, 1.0)),
                                                           (1, Array(3.0, 2.0)),
                                                           (2, Array(2.0, 1.0))))

// Create DDMs
val ddm1 = new DistributedDenseMatrix(rows1)
val ddm2 = new DistributedDenseMatrix(rows2)

val bdm = ddm1.dot(ddm2)

println(bdm)
```

### Diagonal Dot Product
`ddm1.diagonalDot(ddm2)`

The diagonal of the matrix output by `ddm1.dot(ddm2)`

> `ddm1`: `DistributedDenseMatrix` of dimension _(n, k)_  
> `ddm2`: `DistributedDenseMatrix` of dimension _(n, k)_  
>  
> output: `Breeze` `DenseMatrix` of dimension _(1, k)_

```Scala
// Create rows RDD
val rows1: RDD[(Long, Array[Double])] = sc.parallelize(Seq((0, Array(0.0, 1.0)),
                                                           (1, Array(2.0, 3.0)),
                                                           (2, Array(3.0, 4.0))))
val rows2: RDD[(Long, Array[Double])] = sc.parallelize(Seq((0, Array(1.0, 1.0)),
                                                           (1, Array(3.0, 2.0)),
                                                           (2, Array(2.0, 1.0))))

// Create DDMs
val ddm1 = new DistributedDenseMatrix(rows1)
val ddm2 = new DistributedDenseMatrix(rows2)

val bdm = ddm1.diagonalDot(ddm2)

println(bdm)
```

### Cache internal `RDD`

`ddm.cache()`

Cache the `ddm.rows` `RDD`.

> output: `RDD[(Long, Array[Double])]`, `ddm.rows`

## Distributed Sparse Matrix

### Instance Constructor
`new DistributedDenseMatrix(entries: RDD[(Long, (Long, Double))], n: Long)`  
>`rows`: a key value pair `RDD` with values the rows in the matrix, keys the index of the row  
>`n`: the size of the square matrix

```Scala
import lb.edu.aub.hyrax.DistributedSparseMatrix
// Create an entries RDD
val entries: RDD[(Long, (Long, Double))] = sc.parallelize(Seq((0, (0, 1.0)),
                                                              (1, (1, 2.0)),
                                                              (1, (2, 1.0)),
                                                              (2, (1, 1.0)),
                                                              (2, (2, 3.0))))
// Create a DSM
val dsm = new DistributedSparseMatrix(entries, 3)
```

### Right Multiply by DDM
`dsm * ddm`

> `dsm`: `DistributedSparseMatrix` of dimension _(n, n)_  
> `ddm`: `DistributedDenseMatrix` of dimension _(n, k)_  
>  
> output: `DistributedDenseMatrix` of dimension _(n, k)_

```Scala
// Create an entries RDD
val entries: RDD[(Long, (Long, Double))] = sc.parallelize(Seq((0, (0, 1.0)),
                                                              (1, (1, 2.0)),
                                                              (1, (2, 1.0)),
                                                              (2, (1, 1.0)),
                                                              (2, (2, 3.0))))
// Create rows RDD
val rows: RDD[(Long, Array[Double])] = sc.parallelize(Seq((0, Array(0.0, 1.0)),
                                                          (1, Array(2.0, 3.0)),
                                                          (2, Array(3.0, 4.0))))

// Create Matrices
val dsm = new DistributedSparseMatrix(entries, 3)
val ddm1 = new DistributedDenseMatrix(rows)

val ddm2 = dsm * ddm1
```

### Print Matrix Statistics
`dsm.printMatStats()`

Print matrix statistics such as:
- sparsity, proportion of entries which are non-zero 
- number of non-zero entries off the block diagonal
-  proportion of nonzero entries which are off the block diagonal 

### Cache internal `RDD`

`ddm.cache()`

Cache the `ddm.rows` `RDD`.

> output: `RDD[(Long, Array[Double])]`, `ddm.rows`
