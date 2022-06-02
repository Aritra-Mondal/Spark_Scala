package com.sundogsoftware.spark

import com.sundogsoftware.spark.MostPopularSuperheroDataset.{SuperHero, SuperHeroNames}
import org.apache.log4j._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{IntegerType, StringType, StructType}

/** Find the superheros with the least connections. */
object MostObsecureSuperheros {
  case class SuperHeroNames(id:Int, name:String)
  case class SuperHero(value:String)

  /** Main function where the action happens */
  def main(args: Array[String]) {
    // Set the log level for only print errors
    Logger.getLogger("org").setLevel(Level.ERROR)

    // Create a SparkSession using every core of the local machine
    val spark = SparkSession
      .builder
      .appName("MostPopularSuperhero")
      .master("local[*]")
      .getOrCreate()

    // Create schema when reading Marvel-names.txt
    val superHeroNamesSchema = new StructType()
      .add("id", IntegerType, nullable = true)
      .add("name", StringType, nullable = true)

    // Build up a hero ID -> name Dataset
    import spark.implicits._
    val names = spark.read
      .schema(superHeroNamesSchema)
      .option("sep", " ")
      .csv("data/Marvel-names.txt")
      .as[SuperHeroNames]

    val lines = spark.read
      .text("data/Marvel-graph.txt")
      .as[SuperHero]

    val connections = lines
      .withColumn("id", split(col("value"), " ")(0))
      .withColumn("connections", size(split(col("value"), " "))-1)
      .groupBy("id").agg(sum("connections").alias("connections"))

    val minConnectionCount = connections.agg(min("connections")).first().getLong(0)

    val minConnection = connections.filter($"connections" === minConnectionCount)

    val minConnectWithNames = minConnection.join(names, "id")

    println("The following characters have only " + minConnectionCount + " connection(s):")
    minConnectWithNames.select("name").show()
  }
}
