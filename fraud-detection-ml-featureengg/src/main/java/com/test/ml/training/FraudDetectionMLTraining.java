package com.test.ml.training;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.storage.StorageLevel;

import static org.apache.spark.sql.functions.*;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class FraudDetectionMLTraining {
    public static void main(String[] args) {

        if (args.length < 2) {
            System.err.println("Usage: SparkDriver <input-dir> <output-dir>");
            System.exit(1);
        }

        String inputDir = args[0];
        String outputDir = args[1];

        SparkSession spark = SparkSession.builder()
                .appName("Large Scale Fraud Prediction Training")
                .config("spark.sql.broadcastTimeout", "1800") // Handling large
                // data joins/broadcasts
                .getOrCreate();

        // Set checkpoint directory to truncate RDD lineage graphs for massive data splits
        spark.sparkContext().setCheckpointDir(outputDir + "/checkpoints");

        StructType schema = buildSchema();

        // 1. Load Data from TrainingData directory
        // Using inferSchema=true for automatic type detection; for 43GB, manual
        // schema is safer but inferSchema works if data is consistent
        // Load data safely with explicit schema
        Dataset<Row> rawData = spark.read()
                .option("header", "true")
                .schema(schema)
                .csv(inputDir + "/*.csv");

        // 2. EDA: Show Summary Statistics
        System.out.println("Summary Statistics for numerical features:");
        rawData.select("VALUEUSD", "CustomerTotalAmount", "MerchantTotalAmount", "IS_FRAUD").summary().show();

        // 3. Drop Non-Predictive/Sensitive Columns and Cast Numeric Strings
        Dataset<Row> cleanedData = rawData.select(
                col("IS_FRAUD").cast("int").alias("label"),
                col("VALUEUSD"), col("TransactionHour"), col("TransactionDayOfWeek"),
                col("TransactionIsWeekend"), col("TransactionMonth"), col("InCityTransaction"),
                col("ACCOUNT_AGE_DAYS").cast("double"), col("AGE").cast("double"), col("CREDIT_SCORE"),
                col("CustomerTransactionCount"), col("CustomerTotalAmount"), col("CustomerAvgAmount"),
                col("CustomerFraudCount"), col("HighestTransactionValue"), col("LowestTransactionValue"),
                col("CustomerFraudRate"), col("CustomerDeviceCount"), col("MerchantTransactionCount"),
                col("MerchantTotalAmount"), col("MerchantAvgAmount"), col("MerchantFraudCount"),
                col("HighestTransactionValueForMerchant"), col("LowestTransactionValueForMerchant"),
                col("MerchantFraudRate"), col("DailyTransValueAvgForMerchant"), col("WeeklyTransValueAvgForMerchant"),
                col("MonthlyTransValueAvgForMerchant"), col("TopFreqMerchantday"), col("LeastFreqMerchantday"),
                col("TopFreqMerchanthour"), col("LeastFreqMerchanthour"), col("CategoryTransactionCount"),
                col("CategoryTotalAmount"), col("CategoryAvgAmount"), col("CategoryFraudCount"),
                col("HighestTransactionValueForCategory"), col("LowestTransactionValueForCategory"),
                col("CategoryFraudRate"), col("CustMerchTransactionCount"), col("CustMerchTotalAmount"),
                col("CustMerchAvgAmount"), col("CustMerchFraudCount"), col("HighestTransactionValueForMerchantByCustomer"),
                col("CustMerchFraudRate"), col("TimeSinceLastTx"), col("CustMerchCatTransactionCount"),
                col("CustMerchCatTotalAmount"), col("CustMerchCatAvgAmount"), col("CustMerchCatFraudCount"),
                col("CustMerchCatFraudRate"),
                col("CATEGORY"), col("TopSpentCategory") // Categorical
        ).na().fill(0);

        // 4. Handle Class Imbalance (Weighting)
        // Fraud is rare; calculate weights to give more importance to the fraud cases (label=1)
        long totalCount = cleanedData.count();
        long fraudCount = cleanedData.filter("label = 1").count();
        double weightForFraud = (double) totalCount / (2.0 * fraudCount);
        double weightForNonFraud = (double) totalCount / (2.0 * (totalCount - fraudCount));

        Dataset<Row> weightedData = cleanedData.withColumn("classWeight",
                when(col("label").equalTo(1), weightForFraud).otherwise(weightForNonFraud));

        // CRITICAL STORAGE OPTIMIZATION: Persist transformed data to stop recalculations across iterations
        weightedData.persist(StorageLevel.MEMORY_AND_DISK_SER());
        weightedData.count(); // Action triggers asynchronous caching ahead of split

        // 5. Categorical Transformations (StringIndexer)
        String[] categoricalCols = {"CATEGORY", "TopSpentCategory"};
        List<PipelineStage> stages = new ArrayList<>();

        for (String colName : categoricalCols) {
            StringIndexer indexer = new StringIndexer()
                    .setInputCol(colName)
                    .setOutputCol(colName + "Index")
                    .setHandleInvalid("keep"); // Handle new categories in test data
            stages.add(indexer);
        }

        // 6. Assemble Features
        String[] numericCols = {
                "VALUEUSD", "TransactionHour", "TransactionDayOfWeek", "TransactionIsWeekend", "TransactionMonth",
                "InCityTransaction", "ACCOUNT_AGE_DAYS", "AGE", "CREDIT_SCORE", "CustomerTransactionCount",
                "CustomerTotalAmount", "CustomerAvgAmount", "CustomerFraudCount", "HighestTransactionValue",
                "LowestTransactionValue", "CustomerFraudRate", "CustomerDeviceCount", "MerchantTransactionCount",
                "MerchantTotalAmount", "MerchantAvgAmount", "MerchantFraudCount", "HighestTransactionValueForMerchant",
                "LowestTransactionValueForMerchant", "MerchantFraudRate", "DailyTransValueAvgForMerchant",
                "WeeklyTransValueAvgForMerchant", "MonthlyTransValueAvgForMerchant", "TopFreqMerchantday",
                "LeastFreqMerchantday", "TopFreqMerchanthour", "LeastFreqMerchanthour", "CategoryTransactionCount",
                "CategoryTotalAmount", "CategoryAvgAmount", "CategoryFraudCount", "HighestTransactionValueForCategory",
                "LowestTransactionValueForCategory", "CategoryFraudRate", "CustMerchTransactionCount",
                "CustMerchTotalAmount", "CustMerchAvgAmount", "CustMerchFraudCount", "HighestTransactionValueForMerchantByCustomer",
                "CustMerchFraudRate", "TimeSinceLastTx", "CustMerchCatTransactionCount", "CustMerchCatTotalAmount",
                "CustMerchCatAvgAmount", "CustMerchCatFraudCount", "CustMerchCatFraudRate",
                "CATEGORYIndex", "TopSpentCategoryIndex"
        };

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(numericCols)
                .setOutputCol("features");
        stages.add(assembler);

        // 7. Train/Test Split
        Dataset<Row>[] splits = weightedData.randomSplit(new double[]{0.8, 0.2}, 42L);
        Dataset<Row> trainingSet = splits[0];
        Dataset<Row> testSet = splits[1];

        // 8. RandomForest Model
        RandomForestClassifier rf = new RandomForestClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setWeightCol("classWeight")
                .setNumTrees(100)
                .setMaxDepth(7)
                .setMaxBins(32)
                .setSeed(42);
        stages.add(rf);

        // 9. Build and Fit Pipeline
        Pipeline pipeline = new Pipeline().setStages(stages.toArray(new PipelineStage[0]));
        PipelineModel model = pipeline.fit(trainingSet);

        // 10. Evaluation
        Dataset<Row> predictions = model.transform(testSet);
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Model Accuracy on Test Set: " + accuracy);

        // 11. Export Model
        try {
            model.write().overwrite().save(outputDir +
                    "/FraudRandomForestModel");
        }catch(IOException e){
            System.out.println("Not able to persist the model");
            e.printStackTrace();
        }
        /*
           Note on ONNX: Spark Java does not support native ONNX export.
           To export to ONNX for production inference, use the 'onnxmltools' Python library
           or a library like 'MLeap' to serialize the Spark model for cross-platform usage.
        */

        spark.stop();
    }

    private static StructType buildSchema(){

                // 1. Explicit Schema Definition (CRITICAL: Removes double-pass inferSchema overhead)
                StructType schema = new StructType(new StructField[]{
                        new StructField("IS_FRAUD", DataTypes.IntegerType, true, Metadata.empty()),
                        new StructField("VALUEUSD", DataTypes.DoubleType, true, Metadata.empty()),
                        new StructField("TransactionHour", DataTypes.IntegerType, true, Metadata.empty()),
                        new StructField("TransactionDayOfWeek", DataTypes.IntegerType, true, Metadata.empty()),
                        new StructField("TransactionIsWeekend", DataTypes.IntegerType, true, Metadata.empty()),
                        new StructField("TransactionMonth", DataTypes.IntegerType, true, Metadata.empty()),
                        new StructField("InCityTransaction", DataTypes.IntegerType, true, Metadata.empty()),
                        new StructField("ACCOUNT_AGE_DAYS", DataTypes.IntegerType, true
                                , Metadata.empty()),
                        new StructField("AGE", DataTypes.IntegerType, true,
                                Metadata.empty()),
                        new StructField("CREDIT_SCORE", DataTypes.IntegerType, true,
                                Metadata.empty()),
                        new StructField("CustomerTransactionCount",
                                DataTypes.IntegerType, true, Metadata.empty()),
                        new StructField("CustomerTotalAmount", DataTypes.DoubleType, true, Metadata.empty()),
                        new StructField("CustomerAvgAmount", DataTypes.DoubleType, true, Metadata.empty()),
                        new StructField("CustomerFraudCount", DataTypes.IntegerType,
                                true, Metadata.empty()),
                        new StructField("HighestTransactionValue", DataTypes.DoubleType, true, Metadata.empty()),
                        new StructField("LowestTransactionValue", DataTypes.DoubleType, true, Metadata.empty()),
                        new StructField("CustomerFraudRate", DataTypes.DoubleType, true, Metadata.empty()),
                        new StructField("CustomerDeviceCount", DataTypes.IntegerType, true, Metadata.empty()),
                        new StructField("MerchantTransactionCount",
                                DataTypes.IntegerType, true, Metadata.empty()),
                        new StructField("MerchantTotalAmount", DataTypes.DoubleType, true, Metadata.empty()),
                        new StructField("MerchantAvgAmount", DataTypes.DoubleType, true, Metadata.empty()),
                        new StructField("MerchantFraudCount", DataTypes.IntegerType,
                                true, Metadata.empty()),
                        new StructField("HighestTransactionValueForMerchant", DataTypes.DoubleType, true, Metadata.empty()),
                        new StructField("LowestTransactionValueForMerchant", DataTypes.DoubleType, true, Metadata.empty()),
                        new StructField("MerchantFraudRate", DataTypes.DoubleType, true, Metadata.empty()),
                        new StructField("DailyTransValueAvgForMerchant", DataTypes.DoubleType, true, Metadata.empty()),
                        new StructField("WeeklyTransValueAvgForMerchant", DataTypes.DoubleType, true, Metadata.empty()),
                        new StructField("MonthlyTransValueAvgForMerchant", DataTypes.DoubleType, true, Metadata.empty()),
                        new StructField("TopFreqMerchantday", DataTypes.IntegerType, true, Metadata.empty()),
                        new StructField("LeastFreqMerchantday", DataTypes.IntegerType, true, Metadata.empty()),
                        new StructField("TopFreqMerchanthour", DataTypes.IntegerType, true, Metadata.empty()),
                        new StructField("LeastFreqMerchanthour", DataTypes.IntegerType, true, Metadata.empty()),
                        new StructField("CategoryTransactionCount",
                                DataTypes.IntegerType, true, Metadata.empty()),
                        new StructField("CategoryTotalAmount", DataTypes.DoubleType, true, Metadata.empty()),
                        new StructField("CategoryAvgAmount", DataTypes.DoubleType, true, Metadata.empty()),
                        new StructField("CategoryFraudCount", DataTypes.IntegerType,
                                true, Metadata.empty()),
                        new StructField("HighestTransactionValueForCategory", DataTypes.DoubleType, true, Metadata.empty()),
                        new StructField("LowestTransactionValueForCategory", DataTypes.DoubleType, true, Metadata.empty()),
                        new StructField("CategoryFraudRate", DataTypes.DoubleType, true, Metadata.empty()),
                        new StructField("CustMerchTransactionCount",
                                DataTypes.IntegerType, true, Metadata.empty()),
                        new StructField("CustMerchTotalAmount", DataTypes.DoubleType, true, Metadata.empty()),
                        new StructField("CustMerchAvgAmount", DataTypes.DoubleType, true, Metadata.empty()),
                        new StructField("CustMerchFraudCount", DataTypes.IntegerType,
                                true, Metadata.empty()),
                        new StructField("HighestTransactionValueForMerchantByCustomer", DataTypes.DoubleType, true, Metadata.empty()),
                        new StructField("CustMerchFraudRate", DataTypes.DoubleType, true, Metadata.empty()),
                        new StructField("TimeSinceLastTx", DataTypes.DoubleType, true, Metadata.empty()),
                        new StructField("CustMerchCatTransactionCount",
                                DataTypes.IntegerType, true, Metadata.empty()),
                        new StructField("CustMerchCatTotalAmount", DataTypes.DoubleType, true, Metadata.empty()),
                        new StructField("CustMerchCatAvgAmount", DataTypes.DoubleType, true, Metadata.empty()),
                        new StructField("CustMerchCatFraudCount", DataTypes.IntegerType
                                , true, Metadata.empty()),
                        new StructField("CustMerchCatFraudRate", DataTypes.DoubleType, true, Metadata.empty()),
                        new StructField("CATEGORY", DataTypes.StringType, true, Metadata.empty()),
                        new StructField("TopSpentCategory",
                                DataTypes.StringType, true, Metadata.empty()),
                        new StructField("ID",
                                DataTypes.IntegerType, true, Metadata.empty()),
                        new StructField("STATUS",
                                DataTypes.StringType, true, Metadata.empty()),
                        new StructField("TRANS_TIMESTAMP",
                                DataTypes.TimestampType, true, Metadata.empty()),
                        new StructField("CUSTOMER_ID",
                                DataTypes.IntegerType, true, Metadata.empty()),
                        new StructField("DEVICE_ID",
                                DataTypes.IntegerType, true, Metadata.empty()),
                        new StructField("CITY",
                                DataTypes.StringType, true, Metadata.empty()),
                        new StructField("FIRST_NAME",
                                DataTypes.StringType, true, Metadata.empty()),
                        new StructField("LAST_NAME",
                                DataTypes.StringType, true, Metadata.empty()),
                        new StructField("NAME",
                                DataTypes.StringType, true, Metadata.empty()),
                        new StructField("MERCHANT_ID",
                                DataTypes.IntegerType, true, Metadata.empty())
                });



                return schema;

            }
}