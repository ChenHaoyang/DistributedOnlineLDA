import com.typesafe.sbt.SbtScalariform

organization := "com.mad"
name         := "distributedLDA"
version      := "1.0.0"

scalaVersion       := "2.10.6"
crossScalaVersions := Seq("2.10.6", "2.11.8")
javaOptions        ++= Seq("-Xmx9G", "-Xms256M")

resolvers += "Apache-HBase-Spark-snapshots" at "https://repository.apache.org/content/repositories/snapshots"

libraryDependencies ++= Seq(
  ("org.apache.spark" %% "spark-core" % "1.3.0").
    exclude("org.mortbay.jetty", "servlet-api").
    exclude("commons-beanutils", "commons-beanutils-core").
    exclude("commons-collections", "commons-collections").
    exclude("commons-logging", "commons-logging").
    exclude("com.esotericsoftware.minlog", "minlog")
  ,
  "org.apache.spark" %% "spark-mllib" % "1.3.0",
  "org.scalatest" %% "scalatest" % "2.2.4" % Test,
  "org.apache.hbase" % "hbase-spark" % "2.0.0-SNAPSHOT",
  "org.apache.hbase" % "hbase-common" % "2.0.0-SNAPSHOT",
  "org.apache.hbase" % "hbase-server" % "2.0.0-SNAPSHOT",
  "org.apache.hbase" % "hbase-hadoop-compat" % "1.2.2",
  "org.apache.hbase" % "hbase-hadoop2-compat" % "1.2.2"
)

SbtScalariform.defaultScalariformSettings
