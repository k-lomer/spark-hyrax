// TestSuite
// Author: Kyle Lomer

package lb.edu.aub.hyrax


import breeze.numerics.abs
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import LoadData._

@RunWith(classOf[JUnitRunner])
class TestSuite extends FunSuite with BeforeAndAfterAll {


  var spark : SparkSession = _
  var sc : SparkContext = _

  // create const parameters
  val kmax = 200
  val tol = 1e-8

  override def beforeAll(): Unit = {
    // Initialize Spark
    spark = SparkSession
      .builder
      .appName("Aortho TestCase")
      .master("local")
      .getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("WARN")
  }

  override def afterAll(): Unit = {
    sc.stop()
  }

  def maxDiff(m1: RDD[(Long, Array[Double])], m2: RDD[(Long, Array[Double])]): (Long, Double) = {
    m1.join(m2)
      .mapValues(v => v._1.zip(v._2).map(x => x._1 - x._2))
      .mapValues(_.map(abs(_)).max)
      .reduce((x,y) => if (x._2 > y._2) x else y)
  }


  def test_CGS2AorthoLowCom(dir: String) = {
    // MATLAB matrices are 1-indexed, breeze are 0-indexed
    val beginIn = readIntArray(dir + "/../beginIn.out").map(_ - 1)
    val endIn = readIntArray(dir + "/../endIn.out").map(_ - 1)
    val p = new FixedRangePartitioner(beginIn zip endIn)

    val W = new DistributedDenseMatrix(readDenseMatRDD(dir + "/W.out", sc).partitionBy(p))
    val AW = new DistributedDenseMatrix(readDenseMatRDD(dir + "/AW.out", sc).partitionBy(p))
    val Q = new DistributedDenseMatrix(readDenseMatRDD(dir + "/Q.out", sc).partitionBy(p))
    val AQ = new DistributedDenseMatrix(readDenseMatRDD(dir + "/AQ.out", sc).partitionBy(p))

    val (newW, newAW) = AorthoLowCom.aorthoBCGSLowCom(Q, AQ, W, AW, true)

    val exp_newW = readDenseMatRDD(dir + "/newW.out", sc)
    val exp_newAW = readDenseMatRDD(dir + "/newAW.out", sc)

    assert(maxDiff(newW.rows, exp_newW)._2 < 5e-6, "compare matrix W")
    assert(maxDiff(newAW.rows, exp_newAW)._2 < 5e-6, "compare matrix AW")

    assert( Q.dot(newAW).data.forall(abs(_) < 1e-8), "check columns are Aorthogonal")
  }

  def test_CGS2AorthoHighCom(dir: String) = {
    // MATLAB matrices are 1-indexed, breeze are 0-indexed
    val beginIn = readIntArray(dir + "/../beginIn.out").map(_ - 1)
    val endIn = readIntArray(dir + "/../endIn.out").map(_ - 1)
    val p = new FixedRangePartitioner(beginIn zip endIn)

    val W = new DistributedDenseMatrix(readDenseMatRDD(dir + "/W.out", sc).partitionBy(p))
    val Q = new DistributedDenseMatrix(readDenseMatRDD(dir + "/Q.out", sc).partitionBy(p))
    val n = W.dim._1
    val A = new DistributedSparseMatrix(readSparseMatRDD(dir + "/../A.out", sc, false, false).partitionBy(p), n)

    val newW = Aortho.aorthoBCGS(Q, W, A, true)
    val exp_newW = readDenseMatRDD(dir + "/newW.out", sc)

    assert(maxDiff(newW.rows, exp_newW)._2 < 5e-6, "compare matrix W")

    assert( Q.dot(A * newW).data.forall(abs(_) < 1e-8), "check columns are Aorthogonal")
  }


  // CGS2 Aorthonormalization with WREC tests
  for {
    mat <- List("matvf2dNH100100c2", "matvf3dSKY202020c2")
    testCase <- List("2", "20", "50")
  } {
    test("aortho routine, " + mat + "," + testCase) {
      test_CGS2AorthoLowCom("test_data/aortho/" + mat + "/" + testCase)
    }
  }

    // CGS2 Aorthonormalization No WREC tests
  for {
    mat <- List("matvf2dNH100100c2", "matvf3dSKY202020c2")
    testCase <- List("2", "20", "50")
  } {
    test("aortho routine no wrec, " + mat + "," + testCase) {
      test_CGS2AorthoHighCom("test_data/aortho/" + mat + "/" + testCase)
    }
  }

}
