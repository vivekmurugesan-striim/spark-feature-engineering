package com.test.ml.training;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;

import static com.test.ml.training.DataPreprocessor.buildSchemaForRawData;
import static org.apache.spark.sql.functions.*;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class FraudDetectionMLLogisticRegressionClassifier {
    public static void main(String[] args) {

        if (args.length < 2) {
            System.err.println("Usage: SparkDriver <input-dir> <output-dir>");
            System.exit(1);
        }

        String inputDir = args[0];
        String outputDir = args[1];

        SparkSession spark = SparkSession.builder()
                .appName("Large Scale Fraud Prediction Logistic Regression Training")
                .config("spark.sql.broadcastTimeout", "1800")
                .getOrCreate();

        // Set checkpoint directory to truncate RDD lineage graphs for massive data splits
        spark.sparkContext().setCheckpointDir(outputDir + "/checkpoints");

        StructType schema = buildSchemaForRawData();

        // 1. Load Data from TrainingData directory
        // Load data safely with explicit schema
        Dataset<Row> rawData = spark.read()
                .option("header", "true")
                .schema(schema)
                .csv(inputDir + "/*.csv");

        System.out.println("File reading completed successfully..");
        System.out.println("Count of records::" + rawData.count());
        rawData.printSchema();

        // 2. EDA: Show Summary Statistics
        System.out.println("Summary Statistics for numerical features:");
        Dataset<Row> subsetData = rawData.select("VALUEUSD",
                "CustomerTotalAmount", "MerchantTotalAmount", "label");
        subsetData.printSchema();
        List<Row> rows = subsetData.limit(50).takeAsList(50);
        System.out.println("Top 50 rows from the subset.. for summary stats..");
        for (Row row : rows){
            System.out.println("Record::" + row);
        }
        subsetData.summary().show();

        System.out.println("Cleaning data..");

        // 3. Drop Non-Predictive/Sensitive Columns and Cast Numeric Strings
        Dataset<Row> cleanedData = rawData.na().fill(0);

        // 4. Handle Class Imbalance (Weighting)
        long totalCount = cleanedData.count();
        System.out.println("Total count from cleand data::" + totalCount);
        long fraudCount = cleanedData.filter("label = 1").count();
        System.out.println("Fraud count from cleand data::" + fraudCount);
        double weightForFraud = (double) totalCount / (2.0 * fraudCount);
        double weightForNonFraud = (double) totalCount / (2.0 * (totalCount - fraudCount));

        System.out.println("Fraud and non-fraud weights::" + weightForFraud + "::" + weightForNonFraud);

        System.out.println("Generating weighted data::");

        Dataset<Row> weightedData = cleanedData.withColumn("classWeight",
                when(col("label").equalTo(1), weightForFraud).otherwise(weightForNonFraud));

        // CRITICAL STORAGE OPTIMIZATION: Persist transformed data to stop recalculations across iterations
        // Important for large datasets (40GB) to prevent re-reading/re-processing on every pipeline stage
        //weightedData.persist(StorageLevel.MEMORY_AND_DISK_SER());
        long weightedDataCount =
                weightedData.count(); // Action triggers asynchronous caching
        System.out.println("Weighted data count:: " + weightedDataCount);

        // 5. Categorical Transformations (StringIndexer)
        String[] categoricalCols =
                DataPreprocessor.getCategoricalFeatureNames();
        List<PipelineStage> stages = new ArrayList<>();

        for (String colName : categoricalCols) {
            StringIndexer indexer = new StringIndexer()
                    .setInputCol(colName)
                    .setOutputCol(colName + "Index")
                    .setHandleInvalid("keep");
            stages.add(indexer);
        }

        System.out.println("Pipeline stages created..:: ");
        for (PipelineStage stage : stages)
            System.out.println("Stage.. " + stage);

        System.out.println("Assembling features");

        // 6. Assemble Features
        String[] numericCols = DataPreprocessor.getNumericalFeatureNames();

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(numericCols)
                .setOutputCol("features");
        stages.add(assembler);

        // 7. Train/Test Split
        Dataset<Row>[] splits = weightedData.randomSplit(new double[]{0.8, 0.2}, 42L);
        Dataset<Row> trainingSet = splits[0];
        Dataset<Row> testSet = splits[1];

        System.out.println("Train test split done");
        System.out.println("Train data count:: " + trainingSet.count());
        System.out.println("Test data count:: " + testSet.count());

        // 8. LogisticRegression Model (REPLACED RandomForestClassifier)
        LogisticRegression lr = new LogisticRegression()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setWeightCol("classWeight") // Use the calculated weights for imbalance handling
                .setMaxIter(20)
                .setRegParam(0.1) // Default L2 regularization strength
                .setElasticNetParam(0.0); // 0.0 for L2 regularization

        stages.add(lr);

        System.out.println("Starting model training..");

        // 9. Build and Fit Pipeline
        Pipeline pipeline = new Pipeline().setStages(stages.toArray(new PipelineStage[0]));
        PipelineModel model = pipeline.fit(trainingSet);

        System.out.println("Model fit done.. eval started..");

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
                    "/FraudLogisticRegressionModel"); // Updated model path
        }catch(IOException e){
            System.out.println("Not able to persist the model");
            e.printStackTrace();
        }

       /*
          Note on ONNX: Spark Java does not support native ONNX export.
          To export to ONNX for production inference, use the 'onnxmltools' Python library
          or a library like 'MLeap' to serialize the Spark model for cross-platform usage.
       */

        //spark.stop();
    }


}
