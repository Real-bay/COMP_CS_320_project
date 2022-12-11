package assignment22

import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.feature.{MinMaxScaler, VectorAssembler}
import org.apache.spark.ml.{Pipeline, PipelineModel, clustering}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, concat, length, udf}
import breeze.plot._
import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.sql.Encoders.scalaDouble
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.avg
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}


class Assignment {

  val spark: SparkSession = SparkSession.builder()
                                        .appName("assignment22")
                                        .config("spark.driver.host", "localhost")
                                        .master("local")
                                        .getOrCreate()

  // Change shuffle partitions
  spark.conf.set("spark.sql.shuffle.partitions",24)


  // Define schemas for data frames
  val schemaD2: StructType = StructType(Seq(StructField("a",DoubleType,nullable = true),StructField("b",DoubleType,nullable = true),StructField("LABEL",StringType,nullable = true)))
  val schemaD3: StructType = StructType(Seq(StructField("a",DoubleType,nullable = true),StructField("b",DoubleType,nullable = true),StructField("c",DoubleType,nullable = true),StructField("LABEL",StringType,nullable = true)))

  // the data frame to be used in tasks 1 and 4
  val dataD2: DataFrame = spark.read.format("csv")
                                    .option("header", "true")
                                    .schema(schemaD2)
                                    .load("data/dataD2.csv")

  // cache the data frame because it will be used multiple times
  dataD2.cache()


  // the data frame to be used in task 2
  val dataD3: DataFrame = spark.read.format("csv")
                                    .option("header", "true")
                                    .schema(schemaD3)
                                    .load("data/dataD3.csv")
  dataD3.cache()

  // the data frame to be used in task 3 (based on dataD2 but containing numeric labels)
  val labelToInteger: UserDefinedFunction = udf((label: String) => if (label == "Fatal") 0 else 1)
  val dataD2WithLabels: DataFrame = dataD2.withColumn("LABEL", labelToInteger(dataD2("LABEL")))
  dataD2WithLabels.cache()

  // Dirty dataFrame
  val dataD2Dirty: DataFrame = spark.read.format("csv")
                                    .option("header", "true")
                                    .schema(schemaD2)
                                    .load("data/dataD2_dirty.csv")


  def filterData(df: DataFrame): DataFrame = {
    if (df.schema.fieldNames.contains("c")) {
      df.filter(col("a").cast(DoubleType).isNotNull
        && col("b").cast(DoubleType).isNotNull
        && col("c").cast(DoubleType).isNotNull
        && col("LABEL").isin("Ok", "Fatal", 0, 1)
      )
    } else {
      df.filter(col("a").cast(DoubleType).isNotNull
        && col("b").cast(DoubleType).isNotNull
        && col("LABEL").isin("Ok", "Fatal", 0, 1)
      )
    }
  }

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
    val k = BDV(scores.map(_._1.toDouble))
    val score = BDV(scores.map(_._2))

    plt += plot(k, score)
    plt.xlabel = "k"
    plt.ylabel = "Silhouette score"
    plt.title = "Silhouette score for k"

    fig.refresh()
    Thread.sleep(5000)
  }

   // ML-pipeline
  val featureCreator: VectorAssembler = new VectorAssembler()
    .setInputCols(Array("a", "b"))
    .setOutputCol("features")

  val featureScaler: MinMaxScaler = new MinMaxScaler()
    .setInputCol("features")
    .setOutputCol("scaledFeatures")

  val modelFitter: KMeans = new KMeans()
    .setK(2)
    .setSeed(1)
    .setFeaturesCol("scaledFeatures")

  val pipeline: Pipeline = new Pipeline().setStages(Array(featureCreator, featureScaler, modelFitter))


  def task1(df: DataFrame, k: Int): Array[(Double, Double)] = {

    // Filter the DataFrame
    val filteredDf = filterData(df)


    // get min and max values for a and b in df
    val minA = getMin(filteredDf, "a")
    val maxA = getMax(filteredDf, "a")
    val minB = getMin(filteredDf, "b")
    val maxB = getMax(filteredDf, "b")


    // Add params for the pipeline
    val params = ParamMap().put(modelFitter.k, k).put(modelFitter.featuresCol, "scaledFeatures")

    // Run the pipeline
    val model = pipeline.fit(filteredDf, params)
    val scaledDf = model.transform(filteredDf)


    // create the k-means model,get the centers, de-normalize them and return them
    model.stages(2).asInstanceOf[KMeansModel]
      .clusterCenters
      .map(x => (deNormalize(x(0), minA, maxA), deNormalize(x(1), minB, maxB)))

  }

  def task2(df: DataFrame, k: Int): Array[(Double, Double, Double)] = {

    // Filter the DataFrame
    val filteredDf = filterData(df)

    // get min and max values for a, b and c in df
    val minA = getMin(filteredDf, "a")
    val maxA = getMax(filteredDf, "a")
    val minB = getMin(filteredDf, "b")
    val maxB = getMax(filteredDf, "b")
    val minC = getMin(filteredDf, "c")
    val maxC = getMax(filteredDf, "c")

    // Add params for the pipeline
    val params = ParamMap().put(featureCreator.inputCols, Array("a", "b", "c"))
                           .put(featureCreator.outputCol, "unscaledFeatures")
                           .put(featureScaler.inputCol, "unscaledFeatures")
                           .put(featureScaler.outputCol, "features")
                           .put(modelFitter.k, k)
                           .put(modelFitter.featuresCol, "features")

    // Run the pipeline
    val model = pipeline.fit(filteredDf, params)
    val scaledDf = model.transform(filteredDf)

    // Get the centers, de-normalize them and return them
    model.stages(2).asInstanceOf[KMeansModel].clusterCenters
      .map(x => (deNormalize(x(0), minA, maxA), deNormalize(x(1), minB, maxB), deNormalize(x(2), minC, maxC)))
  }


  def task3(df: DataFrame, k: Int): Array[(Double, Double)] = {

    // Filter the DataFrame
    val filteredDf = filterData(df)

    // get min and max values for a and b in df
    val minA = getMin(filteredDf, "a")
    val maxA = getMax(filteredDf, "a")
    val minB = getMin(filteredDf, "b")
    val maxB = getMax(filteredDf, "b")

    // Add params for the pipeline
    val params = ParamMap().put(featureCreator.outputCol, "unscaledFeatures")
                           .put(featureScaler.inputCol, "unscaledFeatures")
                           .put(featureScaler.outputCol, "features")
                           .put(modelFitter.k, k)
                           .put(modelFitter.featuresCol, "features")

    // Run the pipeline with the params
    val model = pipeline.fit(filteredDf,params)


    // Make predictions and select two clusters with the highest number of fatal cases
    val fatalClusters = model.transform(filteredDf)
      .filter(col("LABEL") === 0)
      .groupBy("prediction")
      .count()
      .orderBy(col("count").desc)
      .take(2)
      .map(x => x(0).asInstanceOf[Int])

    // de-normalize the centers and return the correct ones
    Array(model.stages(2).asInstanceOf[KMeansModel].clusterCenters(fatalClusters(0)), model.stages(2).asInstanceOf[KMeansModel].clusterCenters(fatalClusters(1)))
      .map(x => (deNormalize(x(0), minA, maxA), deNormalize(x(1), minB, maxB)))
  }


  def getSilhouetteScore(df: DataFrame, k: Int, max: Int): Array[(Int, Double)] = {

    if (k > max) {
      return Array[(Int, Double)]()
    }

    // create the k-means model and make predictions
    val predictions = new KMeans()
      .setK(k)
      .setSeed(1)
      .fit(df)
      .transform(df)

    // Evaluate clustering by computing Silhouette score
    val score = new ClusteringEvaluator().evaluate(predictions)

    // add score to scores array
    Array.concat(Array((k, score)), getSilhouetteScore(df, k + 1, max))
  }


  // Parameter low is the lowest k and high is the highest one.
  def task4(df: DataFrame, low: Int, high: Int): Array[(Int, Double)]  = {

    // Filter the DataFrame
    val filteredDf = filterData(df)

    // create new data frame with new column "unscaledFeatures"
    val featureDf = new VectorAssembler()
      .setInputCols(Array("a", "b"))
      .setOutputCol("unscaledFeatures")
      .transform(filteredDf)

    // normalize featureDF to [0, 1]
    val scaledData = new MinMaxScaler()
      .setInputCol("unscaledFeatures")
      .setOutputCol("features")
      .fit(featureDf)
      .transform(featureDf)

    val scores = getSilhouetteScore(scaledData, low, high)
    visualizeSilhouetteScore(scores)
    scores
  }

}
