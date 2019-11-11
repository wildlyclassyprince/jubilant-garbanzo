name := "tweet-streamer"

version := "1.0"

scalaVersion := "2.11.0"

val sparkVersion = "2.4.4"

libraryDependencies ++= Seq(
    "org.scalatest" %% "scalatest" % "3.0.5" % "test",
    "org.apache.spark" %% "spark-core" % sparkVersion,
    "org.apache.spark" %% "spark-streaming" % sparkVersion % "provided",
    "org.apache.spark" %% "spark-streaming-twitter" % "1.6.3",
    // https://mvnrepository.com/artifact/org.twitter4j/twitter4j-stream
    "org.twitter4j" % "twitter4j-core" % "4.0.7",
    "org.twitter4j" % "twitter4j-stream" % "4.0.7"

)

// see https://tpolecat.github.io/2017/04/25/scalac-flags.html for scalacOptions descriptions
scalacOptions ++= Seq(
    "-deprecation",     //emit warning and location for usages of deprecated APIs
    "-unchecked",       //enable additional warnings where generated code depends on assumptions
    "-explaintypes",    //explain type errors in more detail
    "-Ywarn-dead-code", //warn when dead code is identified
    "-Xfatal-warnings"  //fail the compilation if there are any warnings
)


