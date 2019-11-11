import scala.io.Source
import java.io.{FileNotFoundException,IOException}

import org.apache.spark.storage.StorageLevel

import org.apache.log4j.Level
import org.apache.spark._
import org.apache.spark.streaming._
import org.apache.spark.streaming.twitter._

import twitter4j.TwitterFactory
import twitter4j.auth.OAuthAuthorization
import twitter4j.conf.ConfigurationBuilder

object TwitterStreamTweets {

    def main(args: Array[String]): Unit = {
        
        // Twitter credentials
        try {
            val configFile = "/home/dorutan/Documents/datascience/github.com/jubilant-garbanzo/tweet-streamer/src/main/resources/config.txt"
            val consumerKey :: consumerSecret :: accessToken :: accessTokenSecret :: _ = Source.fromFile(configFile).getLines.toList

            // Spark configuration and streaming contexts
            val sConfig = new SparkConf().setMaster("local[*]").setAppName("Twitter Streaming Local Popular HashTags & Topics")
            val sContext = new SparkContext(sConfig)
            val sStrContext = new StreamingContext(sContext, Seconds(5))

            // Configure credentials
            val configBuilder = new ConfigurationBuilder
            configBuilder.setDebugEnabled(true)
                .setOAuthConsumerKey(consumerKey)
                .setOAuthConsumerSecret(consumerSecret)
                .setOAuthAccessToken(accessToken)
                .setOAuthAccessTokenSecret(accessTokenSecret)
            
            // Authenticate
            val auth = new OAuthAuthorization(configBuilder.build)

            // Create stream
            val stream = TwitterUtils.createStream(sStrContext, Some(auth))
            val tweets = stream.filter(_.getLang() == "en")

            // Obtain top hashtags
            val hashTags = tweets.flatMap(status => status.getText.split(" ").filter(_.startsWith("#")))
            val topHashTags = hashTags.map(word => (word, 1))
                .reduceByKeyAndWindow(_ + _, Seconds(10))
                .map { case (topic, count) => (count, topic) }
                .transform(_.sortByKey(false))

            topHashTags.foreachRDD(rdd => {
                val topList = rdd.take(10)
                println("\nPopular topics in last 10 seconds (%s total):".format(rdd.count()))
                topList.foreach { case (count, tag) => println("%s (%s tweets)".format(tag, count)) }
            })

            sStrContext.start()
            sStrContext.awaitTermination()

        } catch {
            case e: FileNotFoundException => println("Couldn't find the file.")
            case e: IOException =>println("Got an IOException!")
        }   
    }
}