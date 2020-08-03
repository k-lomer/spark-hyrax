package lb.edu.aub.hyrax

import breeze.linalg.{DenseMatrix => BDM}
import org.apache.spark.rdd.RDD


/* A Distributed Dense Matrix class
// input rows: RDD[(Long, Array[Double])], the index of the row and the row values of the matrix
// input dim: (Long, Long), the shape of the matrix as (rows, cols)
 */
class DistributedDenseMatrix(val rows: RDD[(Long, Array[Double])], val dim: (Long, Long)) extends java.io.Serializable {
  // auxilliary constructor without dim given
  // work out the dim from the RDD
  def this(rows: RDD[(Long, Array[Double])]) = this(rows, (rows.count, rows.first()._2.length))

  def cache() = this.rows.cache()

  // the dot product of the cols of this matrix and another
  def dot(that: DistributedDenseMatrix): BDM[Double] = {
    require(this.dim._1 == that.dim._1, "matrices must have the same number of rows")
    def outerProduct(V: (Array[Double], Array[Double])): BDM[Double] = {
      new BDM[Double](V._1.size, V._2.size, for (c <- V._2; r <- V._1) yield r * c)
    }

    this.rows.join(that.rows)
    .mapValues(outerProduct)
    .values
    .reduce(_ + _)
  }

  // the diagonal only of the dot product of the cols of this matrix and another
  def diagonalDot(that: DistributedDenseMatrix): BDM[Double] = {
    require(this.dim == that.dim, "matrices must have the same dimensions")

    def elementwiseProduct(V: (Array[Double], Array[Double])): BDM[Double] = {
      new BDM[Double](1, V._2.length, V._1.zip(V._2).map{case (a, b) => a * b})
    }
    rows.join(that.rows)
      .mapValues(elementwiseProduct)
      .values
      .reduce(_ + _)
  }

  def +(that: DistributedDenseMatrix): DistributedDenseMatrix = {
    require(this.dim._1 == that.dim._1 && this.dim._2 == that.dim._2, "matrices must have the same dimensions")
    def addArray(v1: Array[Double], v2: Array[Double]): Array[Double] = {
      v1.zip(v2)
        .map(x => x._1 + x._2)
    }
    new DistributedDenseMatrix(this.rows.join(that.rows)
                                        .mapValues(v => addArray(v._1, v._2)), dim)
  }

  def -(that: DistributedDenseMatrix): DistributedDenseMatrix = {
    require(this.dim._1 == that.dim._1 && this.dim._2 == that.dim._2, "matrices must have the same dimensions")
    def subtractArray(v1: Array[Double], v2: Array[Double]): Array[Double] = {
      v1.zip(v2)
        .map(x => x._1 - x._2)
    }
    new DistributedDenseMatrix(this.rows.join(that.rows)
                                        .mapValues(v => subtractArray(v._1, v._2)), dim)
  }

  // Multiply this DDM by a local BDM
  def *(bdm: BDM[Double]): DistributedDenseMatrix = {
    require(this.dim._2 == bdm.rows, "matrices must have suitable dimensions for multiplication")

    val cols = for (i <- 0 until bdm.cols) yield bdm(::, i).toArray

    def multiplyRow(row: Array[Double]): Array[Double] = {
      cols.map(c => c.zip(row)
        .map(x => x._1 * x._2)
        .sum)
        .toArray
    }

    new DistributedDenseMatrix(this.rows.mapValues(multiplyRow), (this.dim._1, bdm.cols))
  }

  // Divide all elements in this DDM by x
  def /(x: Double): DistributedDenseMatrix = {
    new DistributedDenseMatrix(this.rows.mapValues(r => r.map(e => e / x)), this.dim)
  }

  // Divide each column in this DDM by a different factor
  def /(scaleFactors: Array[Double]):  DistributedDenseMatrix = {
    def scaleVector(v: Array[Double]): Array[Double] = {
      v.zip(scaleFactors)
        .map(x => x._1 / x._2)
    }

   new DistributedDenseMatrix(this.rows.mapValues(scaleVector), this.dim)
  }

}
