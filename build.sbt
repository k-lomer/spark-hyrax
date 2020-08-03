name := "hyrax"

version := "1.0"

scalaVersion := "2.11.12"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.3.2"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.3.2"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.3.2"
libraryDependencies += "org.scalanlp" %% "breeze" % "0.12"

libraryDependencies += "junit" % "junit" % "4.10" % Test
libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.5"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.5" % "test"
libraryDependencies += "ch.cern.sparkmeasure" %% "spark-measure" % "0.15"