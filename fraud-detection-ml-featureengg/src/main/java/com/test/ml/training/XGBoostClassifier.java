package com.test.ml.training;


import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import static org.apache.spark.sql.functions.*;

// Import XGBoost library classes
//import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class XGBoostClassifier {
    public static void main(String[] args) {

        if (args.length < 2) {
            System.err.println("Usage: SparkDriver <input-dir> <output-dir>");
            System.exit(1);
        }

        String inputDir = args[0];
        String outputDir = args[1];

        SparkSession spark = SparkSession.builder()
                .appName("Large Scale Fraud Prediction Training")
                .config("spark.sql.broadcastTimeout", "1200")
                .getOrCreate();

        // 1. Load Data from TrainingData directory
        Dataset<Row> rawData = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
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
                col("CATEGORY"), col("TopSpentCategory")
        ).na().fill(0);

        // 4. Handle Class Imbalance (Weighting)
        long totalCount = cleanedData.count();
        long fraudCount = cleanedData.filter("label = 1").count();
        double weightForFraud = (double) totalCount / (2.0 * fraudCount);
        double weightForNonFraud = (double) totalCount / (2.0 * (totalCount - fraudCount));

        Dataset<Row> weightedData = cleanedData.withColumn("classWeight",
                when(col("label").equalTo(1), weightForFraud).otherwise(weightForNonFraud));

        // 5. Categorical Transformations (StringIndexer)
        String[] categoricalCols = {"CATEGORY", "TopSpentCategory"};
        List<PipelineStage> stages = new ArrayList<>();

        for (String colName : categoricalCols) {
            StringIndexer indexer = new StringIndexer()
                    .setInputCol(colName)
                    .setOutputCol(colName + "Index")
                    .setHandleInvalid("keep");
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

        // 8. XGBoost Model Replacement
        // Adjust numWorkers based on your Spark cluster configuration (e.g., total executor cores)
        /*XGBoostClassifier xgb = new XGBoostClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setWeightCol("classWeight") // Uses the exact sample weighting column
                .setNumRound(100)            // Equivalent to setNumTrees
                .setMaxDepth(10)
                .setSeed(42)
                .setObjective("binary:logistic") // Best fit for binary Fraud vs No-Fraud prediction
                .setNumWorkers(2);           // CRITICAL: Set this to match your distributed execution setup
        stages.add(xgb); */

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
            model.write().overwrite().save(outputDir + "/FraudXGBoostModel");
        } catch(IOException e) {
            System.err.println("Failed to save the model: " + e.getMessage());
        }
    }
}
