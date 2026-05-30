package com.test.ml.training;

import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.Arrays;
import java.util.stream.Stream;

import static org.apache.spark.sql.functions.col;

public class DataPreprocessor {

    // Dropping 10 raw columns
    private static final Column[] rawColumnsToDrop = {
            col("ID"),
            col("STATUS"),
            col("TRANS_TIMESTAMP"),
            col("CUSTOMER_ID"),
            col("DEVICE_ID"),
            col("MERCHANT_ID"),
            col("CITY"),
            col("FIRST_NAME"),
            col("LAST_NAME"),
            col("NAME")

    };

    // Dropping 27 features
    private static final Column[] featuresToDrop = {
            col("AGE"),
            col("CREDIT_SCORE"),
            col("CustomerTransactionCount"),
            col("CustomerTotalAmount"),
            col("CustomerFraudCount"),
            col("LowestTransactionValue"),
            col("MerchantTransactionCount"),
            col("MerchantTotalAmount"),
            col("MerchantFraudCount"),
            col("LowestTransactionValueForMerchant"),
            col("DailyTransValueAvgForMerchant"),
            col("WeeklyTransValueAvgForMerchant"),
            col("MonthlyTransValueAvgForMerchant"),
            col("TopFreqMerchantday"),
            col("LeastFreqMerchantday"),
            col("TopFreqMerchanthour"),
            col("LeastFreqMerchanthour"),
            col("CustMerchTransactionCount"),
            col("CustMerchTotalAmount"),
            col("CustMerchFraudCount"),
            col("CategoryTransactionCount"),
            col("CategoryTotalAmount"),
            col("CategoryFraudCount"),
            col("LowestTransactionValueForCategory"),
            col("CustMerchCatTransactionCount"),
            col("CustMerchCatTotalAmount"),
            col("CustMerchCatFraudCount")
    };

    // In total 23 numerical features + 1 label column to be used
    public static final Column[] numericalFeatures = {
            col("IS_FRAUD").cast("int").alias("label"),
            col("VALUEUSD"),
            col("TransactionHour"),
            col("TransactionDayOfWeek"),
            col("TransactionIsWeekend"),
            col("TransactionMonth"),
            col("InCityTransaction"),
            col("ACCOUNT_AGE_DAYS"),
            col("CustomerAvgAmount"),
            col("HighestTransactionValue"),
            col("CustomerFraudRate"),
            col("CustomerDeviceCount"),
            col("MerchantAvgAmount"),
            col("HighestTransactionValueForMerchant"),
            col("MerchantFraudRate"),
            col("CategoryAvgAmount"),
            col("HighestTransactionValueForCategory"),
            col("CategoryFraudRate"),
            col("CustMerchAvgAmount"),
            col("HighestTransactionValueForMerchantByCustomer"),
            col("CustMerchFraudRate"),
            col("TimeSinceLastTx"),
            col("CustMerchCatAvgAmount"),
            col("CustMerchCatFraudRate")
    };

    // Two categorical features to be used
    public static Column[] categoricalFeatures = { col("CATEGORY"), col(
            "TopSpentCategory") };

    public static void main(String[] args){

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

        StructType schema = buildSchemaForData();

        // 1. Load Data from TrainingData directory
        // Using inferSchema=true for automatic type detection; for 43GB, manual
        // schema is safer but inferSchema works if data is consistent
        // Load data safely with explicit schema
        Dataset<Row> rawData = spark.read()
                .option("header", "true")
                .schema(schema)
                .csv(inputDir + "/*.csv");

        System.out.println("File reading completed successfully..");
        /*System.out.println("Count of records::" + rawData.count());
        rawData.printSchema();*/

        Dataset<Row> updatedDataset = selectRequiredColumns(rawData);

        updatedDataset.write()
                .mode("overwrite")
                .option("header", "true")
                .csv(outputDir);

        System.out.println("Training record preprocessing completed..");

        spark.stop();

    }

    private static Dataset<Row> selectRequiredColumns(Dataset<Row> rawData){

        System.out.println("Selecting required columns.. from the raw " +
                "features..");

        Column[] combinedFeatures = Stream.concat(Arrays.stream(numericalFeatures), Arrays.stream(categoricalFeatures))
                .toArray(Column[]::new);

        Dataset<Row> result =
                rawData.select(combinedFeatures);

        int columnCount = result.columns().length;
        System.out.println(".. Selected column count::" + columnCount);
        long resultRowCount = result.count();
        System.out.println(".. selected row count::" + resultRowCount);

        System.out.println("Schema of the selected features + label::");
        result.printSchema();

        return result;
    }



    // The buildSchema method is necessary for explicit schema loading
    public static StructType buildSchemaForData(){

        // 1. Explicit Schema Definition (CRITICAL: Removes double-pass inferSchema overhead)
        StructType schema = new StructType(new StructField[]{
                new StructField("ID", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("STATUS", DataTypes.StringType, true, Metadata.empty()),
                new StructField("TRANS_TIMESTAMP", DataTypes.TimestampType, true, Metadata.empty()),
                new StructField("VALUEUSD", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("CUSTOMER_ID", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("DEVICE_ID", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("MERCHANT_ID", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("CATEGORY", DataTypes.StringType, true, Metadata.empty()),
                new StructField("TransactionHour", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("TransactionDayOfWeek", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("TransactionIsWeekend", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("TransactionMonth", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("InCityTransaction", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("ACCOUNT_AGE_DAYS", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("AGE", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("CITY", DataTypes.StringType, true, Metadata.empty()),
                new StructField("CREDIT_SCORE", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("FIRST_NAME", DataTypes.StringType, true, Metadata.empty()),
                new StructField("LAST_NAME", DataTypes.StringType, true, Metadata.empty()),
                new StructField("CustomerTransactionCount", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("CustomerTotalAmount", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("CustomerAvgAmount", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("CustomerFraudCount", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("HighestTransactionValue", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("LowestTransactionValue", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("CustomerFraudRate", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("CustomerDeviceCount", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("TopSpentCategory", DataTypes.StringType, true, Metadata.empty()),
                new StructField("NAME", DataTypes.StringType, true, Metadata.empty()),
                new StructField("MerchantTransactionCount", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("MerchantTotalAmount", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("MerchantAvgAmount", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("MerchantFraudCount", DataTypes.IntegerType, true, Metadata.empty()),
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
                new StructField("CategoryTransactionCount", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("CategoryTotalAmount", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("CategoryAvgAmount", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("CategoryFraudCount", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("HighestTransactionValueForCategory", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("LowestTransactionValueForCategory", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("CategoryFraudRate", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("CustMerchTransactionCount", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("CustMerchTotalAmount", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("CustMerchAvgAmount", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("CustMerchFraudCount", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("HighestTransactionValueForMerchantByCustomer", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("CustMerchFraudRate", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("TimeSinceLastTx", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("CustMerchCatTransactionCount", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("CustMerchCatTotalAmount", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("CustMerchCatAvgAmount", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("CustMerchCatFraudCount", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("CustMerchCatFraudRate", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("IS_FRAUD", DataTypes.IntegerType, true, Metadata.empty())
        });
        return schema;

    }
}
