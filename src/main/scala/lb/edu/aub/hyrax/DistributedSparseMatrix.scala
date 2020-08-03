package lb.edu.aub.hyrax

import org.apache.spark.rdd.RDD


/* A Distributed Sparse Matrix class
// input entries: RDD[(Long, (Long, Double))], the non-zero matrix entries in alternative COO format (j, (i, v))
// input n: the size of the matrix, note that the matrix is expected to be square
 */
class DistributedSparseMatrix (val entries: RDD[(Long, (Long, Double))], val n: Long) extends java.io.Serializable {

  def cache() = this.entries.cache()

  def addArray(v1: Array[Double], v2: Array[Double]): Array[Double] = {
    v1.zip(v2)
      .map(x => x._1 + x._2)
  }

  def *(ddm: DistributedDenseMatrix): DistributedDenseMatrix = {
    if (ddm.rows.partitioner.isDefined) {
      new DistributedDenseMatrix(
        this.entries.join(ddm.rows)
          .map { case (_, ((i, v), vec)) => (i, vec.map(_ * v)) }
          .reduceByKey(addArray)
          .partitionBy(ddm.rows.partitioner.get), ddm.dim)
    } else {
      new DistributedDenseMatrix(
        this.entries.join(ddm.rows)
          .map { case (_, ((i, v), vec)) => (i, vec.map(_ * v)) }
          .reduceByKey(addArray), ddm.dim)
    }
  }

  def printMatStats() = {
    val total_entries = n.toDouble * n.toDouble
    val sparsity = this.entries.count().toDouble / total_entries
    println("Sparsity: " + sparsity)
    val p = this.entries.partitioner.get
    val outOfDiag = this.entries.filter{case (j, (i, _)) => p.getPartition(j) != p.getPartition(i)}.count
    println("Entries off diagonal block: " + outOfDiag)
    println("Proportion off diagonal block: " + (outOfDiag.toDouble / this.entries.count().toDouble))
  }
}
