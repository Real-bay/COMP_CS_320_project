package assignment22

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.{MinMaxScaler, VectorAssembler}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, concat, length, udf}


class Assignment {

  val spark: SparkSession = SparkSession.builder()
                                        .appName("assignment22")
                                        .config("spark.driver.host", "localhost")
                                        .master("local")
                                        .getOrCreate()

  // the data frame to be used in tasks 1 and 4
  val dataD2: DataFrame = spark.read.format("csv")
                                    .option("header", "true")
                                    .option("inferSchema", "true")
                                    .load("data/dataD2.csv")

  // the data frame to be used in task 2
  val dataD3: DataFrame = spark.read.format("csv")
                                    .option("header", "true")
                                    .option("inferSchema", "true")
                                    .load("data/dataD3.csv")

  // the data frame to be used in task 3 (based on dataD2 but containing numeric labels)
  val labelToInteger: UserDefinedFunction = udf((label: String) => if (label == "Fatal") 0 else 1)
  val dataD2WithLabels: DataFrame = dataD2.withColumn("LABEL", labelToInteger(dataD2("LABEL")))


  def task1(df: DataFrame, k: Int): Array[(Double, Double)] = {

    // get min and max values for a and b in df
    val minA = df.select("a").agg(Map("a" -> "min")).first.getDouble(0)
    val maxA = df.select("a").agg(Map("a" -> "max")).first.getDouble(0)
    val minB = df.select("b").agg(Map("b" -> "min")).first.getDouble(0)
    val maxB = df.select("b").agg(Map("b" -> "max")).first.getDouble(0)

    // create new data frame with new column "features"
    val cols = Array("a", "b")
    val assembler = new VectorAssembler().setInputCols(cols).setOutputCol("features")
    val featureDf = assembler.transform(df.drop("LABEL"))

    // normalize featureDF to [0, 1]
    val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")
    val scalerModel = scaler.fit(featureDf)
    val scaledData = scalerModel.transform(featureDf)

    // create the k-means model
    val kmeans = new KMeans().setK(k).setSeed(1).setFeaturesCol("scaledFeatures")
    val model = kmeans.fit(scaledData)

    // de-normalize the centers and return them
    val centers = model.clusterCenters.map(x => (x(0) * (maxA - minA) + minA, x(1) * (maxB - minB) + minB))
    centers
  }


  def task2(df: DataFrame, k: Int): Array[(Double, Double, Double)] = {

    // get min and max values for a, b and c in df
    val minA = df.select("a").agg(Map("a" -> "min")).first.getDouble(0)
    val maxA = df.select("a").agg(Map("a" -> "max")).first.getDouble(0)
    val minB = df.select("b").agg(Map("b" -> "min")).first.getDouble(0)
    val maxB = df.select("b").agg(Map("b" -> "max")).first.getDouble(0)
    val minC = df.select("c").agg(Map("c" -> "min")).first.getDouble(0)
    val maxC = df.select("c").agg(Map("c" -> "max")).first.getDouble(0)

    // create new data frame with new column "features"
    val cols = Array("a", "b", "c")
    val assembler = new VectorAssembler().setInputCols(cols).setOutputCol("features")
    val featureDf = assembler.transform(df.drop("LABEL"))

    // normalize featureDF to [0, 1]
    val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")
    val scalerModel = scaler.fit(featureDf)
    val scaledData = scalerModel.transform(featureDf)

    // create the k-means model
    val kmeans = new KMeans().setK(k).setSeed(1).setFeaturesCol("scaledFeatures")
    val model = kmeans.fit(scaledData)


    // de-normalize the centers and return them
    val centers = model.clusterCenters.map(x => (x(0) * (maxA - minA) + minA, x(1) * (maxB - minB) + minB, x(2) * (maxC - minC) + minC))
    centers
  }

  def task3(df: DataFrame, k: Int): Array[(Double, Double)] = {

    // get min and max values for a and b in df
    val minA = df.select("a").agg(Map("a" -> "min")).first.getDouble(0)
    val maxA = df.select("a").agg(Map("a" -> "max")).first.getDouble(0)
    val minB = df.select("b").agg(Map("b" -> "min")).first.getDouble(0)
    val maxB = df.select("b").agg(Map("b" -> "max")).first.getDouble(0)

    // create new data frame with new column "features"
    val cols = Array("a", "b")
    val assembler = new VectorAssembler().setInputCols(cols).setOutputCol("features")
    val featureDf = assembler.transform(df)

    // normalize featureDF to [0, 1]
    val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")
    val scalerModel = scaler.fit(featureDf)
    val scaledData = scalerModel.transform(featureDf)

    // create the k-means model
    val kmeans = new KMeans().setK(k).setSeed(1).setFeaturesCol("scaledFeatures")
    val model = kmeans.fit(scaledData)

    // Make predictions
    val predictions = model.transform(scaledData)

    // Select two clusters with the highest number of fatal cases
    val fatalClusters = predictions.filter(predictions("LABEL") === 0).groupBy("prediction").count().orderBy(col("count").desc).take(2).map(x => x(0).asInstanceOf[Int])

    // de-normalize the centers and return the correct ones
    val centers = model.clusterCenters.map(x => (x(0) * (maxA - minA) + minA, x(1) * (maxB - minB) + minB))
    Array(centers(fatalClusters(0)), centers(fatalClusters(1)))
  }

  // Parameter low is the lowest k and high is the highest one.
  def task4(df: DataFrame, low: Int, high: Int): Array[(Int, Double)]  = {

    var scores = Array[(Int, Double)]()

    // create new data frame with new column "features"
    val cols = Array("a", "b")
    val assembler = new VectorAssembler().setInputCols(cols).setOutputCol("features")
    val featureDf = assembler.transform(df.drop("LABEL"))

    // normalize featureDF to [0, 1]
    val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")
    val scalerModel = scaler.fit(featureDf)
    val scaledData = scalerModel.transform(featureDf).drop("features", "LABEL").withColumnRenamed("scaledFeatures", "features")
    scaledData.show()

    // Do this with recursion later!!!
    for (k <- low to high) {

      // create the k-means model
      val kmeans = new KMeans().setK(k).setSeed(1)
      val model = kmeans.fit(scaledData)

      scaledData.show()


      // Make predictions
      val predictions = model.transform(scaledData)

      // Evaluate clustering by computing Silhouette score
      val evaluator = new ClusteringEvaluator()
      val score = evaluator.evaluate(predictions)

      // add score to scores array
      scores = scores :+ (k, score)
    }

    scores
  }

}
