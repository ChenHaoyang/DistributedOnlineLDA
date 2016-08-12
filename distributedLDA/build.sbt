import com.typesafe.sbt.SbtScalariform

organization := "com.mad"
name         := "distributedLDA"
version      := "1.0.0"

scalaVersion       := "2.10.6"
crossScalaVersions := Seq("2.10.6", "2.11.8")
javaOptions        ++= Seq("-Xmx9G", "-Xms256M")

resolvers += "Apache-HBase-Spark-snapshots" at "https://repository.apache.org/content/repositories/snapshots"

libraryDependencies ++= Seq(
  ("org.apache.spark" %% "spark-core" % "1.3.0" % "provided").
    exclude("org.mortbay.jetty", "servlet-api").
    exclude("commons-beanutils", "commons-beanutils-core").
    exclude("commons-collections", "commons-collections").
    exclude("commons-logging", "commons-logging").
    exclude("com.esotericsoftware.minlog", "minlog")
  ,
  "org.apache.spark" %% "spark-mllib" % "1.3.0" % "provided",
  "org.scalatest" %% "scalatest" % "2.2.4" % Test,
  "org.apache.hbase" % "hbase-spark" % "2.0.0-SNAPSHOT",
  "org.apache.hbase" % "hbase-common" % "2.0.0-SNAPSHOT",
  "org.apache.hbase" % "hbase-server" % "2.0.0-SNAPSHOT",
  "org.apache.hbase" % "hbase-hadoop-compat" % "1.2.2",
  "org.apache.hbase" % "hbase-hadoop2-compat" % "1.2.2"
)

assemblyMergeStrategy in assembly := {
  case PathList("javax", "servlet", xs @ _*)         => MergeStrategy.first
  case PathList(ps @ _*) if ps.last endsWith ".thrift" => MergeStrategy.first
  case PathList(ps @ _*) if ps.last endsWith ".html" => MergeStrategy.first
  case PathList(ps @ _*) if ps.last endsWith ".properties" => MergeStrategy.first
  case PathList(ps @ _*) if ps.last endsWith ".xml" => MergeStrategy.first
  case PathList(ps @ _*) if ps.last endsWith ".types" => MergeStrategy.first
  case PathList(ps @ _*) if ps.last endsWith ".class" => MergeStrategy.first
  case "application.conf"                            => MergeStrategy.concat
  case "unwanted.txt"                                => MergeStrategy.discard
  case x =>
    val oldStrategy = (assemblyMergeStrategy in assembly).value
    oldStrategy(x)
}

SbtScalariform.defaultScalariformSettings
