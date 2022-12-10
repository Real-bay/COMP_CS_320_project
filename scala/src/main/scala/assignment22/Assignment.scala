package assignment22

import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.{MinMaxScaler, VectorAssembler}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, concat, length, udf}
import breeze.linalg.DenseVector
import breeze.plot._

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


  def getMin(df: DataFrame, colName: String): Double = {
    df.select(colName).agg(Map(colName -> "min")).first.getDouble(0)
  }

  def getMax(df: DataFrame, colName: String): Double = {
    df.select(colName).agg(Map(colName -> "max")).first.getDouble(0)
  }

  def getFeatureDf(df: DataFrame, featureCols: Array[String]): DataFrame = {
    new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")
      .transform(df)
  }

  def getScaledData(df: DataFrame, colNames: Array[String]): DataFrame = {
    new MinMaxScaler()
    .setInputCol("features")
    .setOutputCol("scaledFeatures")
    .fit(df)
    .transform(df)
      .drop(colNames: _*)
      .withColumnRenamed("scaledFeatures", "features")
  }

  def getModel(df: DataFrame, k: Int): KMeansModel = {
    new KMeans()
      .setK(k)
      .setSeed(1)
      .fit(df)
  }

  def deNormalize(norm: Double, min: Double, max: Double): Double = {
    norm * (max - min) + min
  }

  def visualizeSilhouetteScore(scores: Array[(Int, Double)]): Unit = {
    val fig = Figure()
    val plt = fig.subplot(0)
    val k = DenseVector(scores.map(_._1.toDouble))
    val score = DenseVector(scores.map(_._2))

    plt += plot(k, score)
    plt.xlabel = "k"
    plt.ylabel = "Silhouette score"
    plt.title = "Silhouette score for k"

    fig.refresh()
    Thread.sleep(5000)
  }


  def task1(df: DataFrame, k: Int): Array[(Double, Double)] = {

    // get min and max values for a and b in df
    val minA = getMin(df, "a")
    val maxA = getMax(df, "a")
    val minB = getMin(df, "b")
    val maxB = getMax(df, "b")

    // create new data frame with new column "features"
    val featureDf = getFeatureDf(df, Array("a", "b"))

    // normalize featureDF to [0, 1]
    val scaledData = getScaledData(featureDf, Array("features", "LABEL"))

    // create the k-means model,get the centers, de-normalize them and return them
    getModel(scaledData, k)
      .clusterCenters
      .map(x => (deNormalize(x(0), minA, maxA), deNormalize(x(1), minB, maxB)))
  }


  def task2(df: DataFrame, k: Int): Array[(Double, Double, Double)] = {

    // get min and max values for a, b and c in df
    val minA = getMin(df, "a")
    val maxA = getMax(df, "a")
    val minB = getMin(df, "b")
    val maxB = getMax(df, "b")
    val minC = getMin(df, "c")
    val maxC = getMax(df, "c")

    // create new data frame with new column "features"
    val featureDf = getFeatureDf(df, Array("a", "b", "c"))

    // normalize featureDF to [0, 1]
    val scaledData = getScaledData(featureDf, Array("features", "LABEL"))

    // create the k-means model,get the centers, de-normalize them and return them
    getModel(scaledData, k)
      .clusterCenters
      .map(x => (deNormalize(x(0), minA, maxA), deNormalize(x(1), minB, maxB), deNormalize(x(2), minC, maxC)))
  }


  def task3(df: DataFrame, k: Int): Array[(Double, Double)] = {

    // get min and max values for a and b in df
    val minA = getMin(df, "a")
    val maxA = getMax(df, "a")
    val minB = getMin(df, "b")
    val maxB = getMax(df, "b")

    // create new data frame with new column "features"
    val featureDf = getFeatureDf(df, Array("a", "b"))

    // normalize featureDF to [0, 1]
    val scaledData = getScaledData(featureDf, Array("features"))

    // create the k-means model
    val model = getModel(scaledData, k)

    // Make predictions and select two clusters with the highest number of fatal cases
    val fatalClusters = model
      .transform(scaledData)
      .filter(col("LABEL") === 0)
      .groupBy("prediction")
      .count()
      .orderBy(col("count").desc)
      .take(2)
      .map(x => x(0).asInstanceOf[Int])

    // de-normalize the centers and return the correct ones
    Array(model.clusterCenters(fatalClusters(0)), model.clusterCenters(fatalClusters(1)))
      .map(x => (deNormalize(x(0), minA, maxA), deNormalize(x(1), minB, maxB)))
  }


  def getSilhouetteScore(df: DataFrame, k: Int, max: Int): Array[(Int, Double)] = {

    if (k > max) {
      return Array[(Int, Double)]()
    }

    // create the k-means model and make predictions
    val predictions = getModel(df, k)
      .transform(df)

    // Evaluate clustering by computing Silhouette score
    val score = new ClusteringEvaluator().evaluate(predictions)

    // add score to scores array
    Array.concat(Array((k, score)), getSilhouetteScore(df, k + 1, max))
  }


  // Parameter low is the lowest k and high is the highest one.
  def task4(df: DataFrame, low: Int, high: Int): Array[(Int, Double)]  = {

    // create new data frame with new column "features"
    val featureDf = getFeatureDf(df, Array("a", "b"))

    // normalize featureDF to [0, 1]
    val scaledData = getScaledData(featureDf, Array("features", "LABEL"))

    val scores = getSilhouetteScore(scaledData, low, high)
    visualizeSilhouetteScore(scores)
    scores
  }

}
