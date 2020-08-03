package lb.edu.aub.hyrax

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.io.Source

object LoadData {
  // Read a whitespace separated integer array from file
  // Store as Array[Long]
  def readIntArray(path: String): Array[Long] = {
    scala.io.Source.fromFile(path)
      .mkString.split("\\s+")
      .map(_.toDouble)
      .map(_.toLong)
  }


  // Read a whitespace separated double array from file
  // Store as RDD[(Long, Double)] of (index, value)
  def readDoubleRDD(path: String, sc: SparkContext): RDD[(Long, Double)] ={
    sc.textFile(path)
      .map(_.toDouble)
      .zipWithIndex
      .map(_.swap)
  }

  // Read a sparse matrix from file
  // Each line in the file is whitespace separated row, col, value
  // Store as RDD[(Long, (Long, Double)] of (row, (col, value)) if rowKey, (col, (row, value)) otherwise
  def readSparseMatRDD(path: String, sc: SparkContext, rowKey: Boolean, reflect: Boolean): RDD[(Long, (Long, Double))] = {
    val load = sc.textFile(path)
      .map(_.split("\\s+"))
      .map(xs => if(rowKey) (xs(0).toLong - 1, (xs(1).toLong - 1, xs(2).toDouble)) else (xs(1).toLong - 1, (xs(0).toLong - 1, xs(2).toDouble)))

    if (reflect){
      load.union(load.filter{case (x, (y, _)) => x != y}.map{case(x, (y, v)) => (y, (x, v))})
    } else {
      load
    }
  }

  // Read a dense matrix from file
  // Each line in the file is a whitespace separated row of the matrix
  def readDenseMatRDD(path: String, sc: SparkContext): RDD[(Long, Array[Double])] =sc.textFile(path)
    .map(_.split("\\s+"))
    .map(_.map(_.toDouble))
    .zipWithIndex()
    .map(_.swap)

  def readVal(path: String): Double = Source.fromFile(path).mkString.toDouble
}
