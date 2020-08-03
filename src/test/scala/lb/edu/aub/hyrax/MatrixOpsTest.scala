// MatrixTest
// Author: Kyle Lomer

package lb.edu.aub.hyrax

import breeze.linalg.{DenseMatrix, max}
import breeze.numerics.abs
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{BeforeAndAfterAll, FunSuite}

import scala.io.Source


@RunWith(classOf[JUnitRunner])
class MatrixOpsTest extends FunSuite with BeforeAndAfterAll {


  var spark : SparkSession = _
  var sc : SparkContext = _

  override def beforeAll(): Unit = {
    // Initialize Spark
    spark = SparkSession
      .builder
      .master("local")
      .appName("Matrix Ops TestCase")
      .getOrCreate()
    sc = spark.sparkContext
  }

  override def afterAll(): Unit = {
    sc.stop()
  }

  def seqToDDM(s: Seq[(Long, Array[Double])]): DistributedDenseMatrix = new DistributedDenseMatrix(sc.parallelize(s))

  def seqToDSM(s: Seq[(Long, (Long, Double))], n: Long): DistributedSparseMatrix = new DistributedSparseMatrix(sc.parallelize(s), n)

  def readDenseMat(path: String): DistributedDenseMatrix = new DistributedDenseMatrix(sc.textFile(path)
    .map(_.split("\\s+"))
    .map(_.map(_.toDouble))
    .zipWithIndex()
    .map(_.swap))

  def readVec(path: String): RDD[(Long, Double)] = sc.textFile(path)
    .map(_.toDouble)
    .zipWithIndex()
    .map(_.swap)

  def readArray (path: String): Array[Double] = Source.fromFile(path)
    .getLines
    .map(_.toDouble)
    .toArray

  def checkMatrixValues(m: DistributedDenseMatrix, exp_m: Array[Array[Double]]): Boolean = {
    val data = m.rows.collect
    data.map(r => r._2.sameElements(exp_m(r._1.toInt)))
      .reduce(_ && _)
  }

  def checkVectorValues(v: RDD[(Long, Double)], exp_v: Array[Double]): Boolean = {
    val data = v.collect
    data.map(x => x._2 == exp_v(x._1.toInt))
      .reduce(_ && _)
  }

  test("multiply distributed by local matrix") {
    val dm = DenseMatrix((1.0, 2.0), (3.0, 4.0))
    val rm = seqToDDM(Seq((0.toLong ,Array(0.0, 1.0)), (1.toLong ,Array(2.0, 3.0))))
    assert(checkMatrixValues(rm * dm, Array(Array(3.0, 4.0), Array(11.0, 16.0))))
  }

   test("dot"){
    val rm1 = seqToDDM(Seq((0.toLong ,Array(0.0, 1.0)), (1.toLong ,Array(2.0, 3.0)), (2.toLong ,Array(3.0, 4.0))))
    val rm2 = seqToDDM(Seq((0.toLong ,Array(1.0, 1.0)), (1.toLong ,Array(4.0, 2.0)), (2.toLong ,Array(0.0, 1.0))))

    val expected = DenseMatrix((8.0, 7.0), (13.0, 11.0))

    val res = rm1.dot(rm2)

    assert((res :== expected).data.reduce(_ && _))
  }

  test(" diagonal dot"){
    val rm1 = seqToDDM(Seq((0.toLong ,Array(0.0, 1.0)), (1.toLong ,Array(2.0, 3.0)), (2.toLong ,Array(3.0, 4.0))))
    val rm2 = seqToDDM(Seq((0.toLong ,Array(1.0, 1.0)), (1.toLong ,Array(4.0, 2.0)), (2.toLong ,Array(0.0, 1.0))))

    val expected = DenseMatrix((8.0, 11.0))

    val res = rm1.diagonalDot(rm2)

    assert((res :== expected).data.reduce(_ && _))
  }

  test("dot full"){
    val W = readDenseMat("test_data/W.out")
    val r = readDenseMat("test_data/r.out")

    val alpha = W.dot(r)
    val exp_alpha = DenseMatrix(readArray("test_data/alpha.out")).t

    assert(max(abs(alpha - exp_alpha)) < 1e-8)
  }

  test("subtract matrix"){
    val rm1 = seqToDDM(Seq((0.toLong ,Array(1.0, 2.0)), (1.toLong ,Array(4.0, 6.0))))
    val rm2 = seqToDDM(Seq((0.toLong ,Array(0.0, 1.0)), (1.toLong ,Array(5.0, 3.0))))

    assert(checkMatrixValues(rm1 - rm2, Array(Array(1.0, 1.0), Array(-1.0, 3.0))))
  }

  test("add matrix"){
    val rm1 = seqToDDM(Seq((0.toLong ,Array(1.0, 2.0)), (1.toLong ,Array(4.0, 6.0))))
    val rm2 = seqToDDM(Seq((0.toLong ,Array(0.0, 1.0)), (1.toLong ,Array(5.0, 3.0))))

    assert(checkMatrixValues(rm1 + rm2, Array(Array(1.0, 3.0), Array(9.0, 9.0))))
  }

  test("scale columns by double") {
    val rm = seqToDDM(Seq((0.toLong ,Array(1.0, 2.0)), (1.toLong ,Array(4.0, 6.0))))

    assert(checkMatrixValues(rm / 0.5, Array(Array(2.0, 4.0), Array(8.0, 12.0))))
  }

  test("scale columns by array") {
    val rm = seqToDDM(Seq((0.toLong ,Array(1.0, 2.0)), (1.toLong ,Array(4.0, 6.0))))
    val scale = Array(0.5, 2.0)

    assert(checkMatrixValues(rm / scale, Array(Array(2.0, 1.0), Array(8.0, 3.0))))
  }

  test("multiply sparse matrix by dense matrix") {
    val dsm = seqToDSM(Seq((0,(0, 2.0)), (0,(1, 0.5)), (1,(1, 4.0))), 2)
    val ddm = seqToDDM(Seq((0 ,Array(1.0, 2.0)), (1 ,Array(4.0, 6.0))))

    assert(checkMatrixValues(dsm * ddm, Array(Array(2.0, 4.0), Array(16.5, 25.0))))
  }

}
