// LoadData
// Author: Kyle Lomer
package lb.edu.aub.hyrax

import org.apache.spark.Partitioner

/* A Range Partitioner which allows exact ranges to be specified
// This is useful when the size of the ranges is not uniform
// input ranges: List[(Long, Long)], the begin and end index for each partition
 */
class FixedRangePartitioner(val ranges: Array[(Long, Long)]) extends Partitioner {
  override def numPartitions: Int = ranges.length

  override def getPartition(key: Any): Int = {
    val k: Long = key.asInstanceOf[Long]

    for (i <- ranges.indices){
      if(ranges(i)._1 <= k && ranges(i)._2 >= k) {return i}
    }
    // if out of range then place in partition 0 by default
    return 0
  }

  override def equals(other: scala.Any): Boolean = {
    other match {
      case obj: FixedRangePartitioner => obj.ranges == ranges
      case _ => false
    }
  }
}
