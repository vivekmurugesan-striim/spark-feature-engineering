package com.test.feature.engg;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import static org.apache.spark.sql.functions.*;

public class TrainingRecordGenerator {
    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println("Usage: TrainingRecordGenerator <input-dir> <output-dir>");
            System.exit(1);
        }

        String inputDir = args[0];
        String outputDir = args[1];

        SparkSession spark = SparkSession.builder()
                .appName("Combine Banking Features for Training")
                .getOrCreate();

        // 1. Load the pre-generated Feature Datasets
        Dataset<Row> txFeats = spark.read().option("header", "true").option("inferSchema", "true")
                .csv(inputDir + "/TransactionFeatures/*.csv");

        Dataset<Row> custFeats = spark.read().option("header", "true").option("inferSchema", "true")
                .csv(inputDir + "/CustomerFeatures/*.csv");

        Dataset<Row> merchFeats = spark.read().option("header", "true").option("inferSchema", "true")
                .csv(inputDir + "/MerchantFeatures/*.csv");

        Dataset<Row> merchCatFeats = spark.read().option("header", "true").option("inferSchema", "true")
                .csv(inputDir + "/MerchantCategoryFeatures/*.csv");

        Dataset<Row> custMerchFeats = spark.read().option("header", "true").option("inferSchema", "true")
                .csv(inputDir + "/CustomerMerchantFeatures/*.csv");

        Dataset<Row> custMerchCatFeats = spark.read().option("header", "true").option("inferSchema", "true")
                .csv(inputDir + "/CustomerMerchantCategoryFeatures/*.csv");

        Dataset<Row> fraudTrans = spark.read().option("header", "true").option("inferSchema", "true")
                .csv(inputDir + "/FraudTrans/*.csv");

        // 2. Combine Datasets using TransactionFeatures as the Base
        // We use left joins to ensure every transaction is preserved
        Dataset<Row> combined = txFeats
                // Join with Customer Features
                .join(custFeats, txFeats.col("CUSTOMER_ID").equalTo(custFeats.col("CF_ID")), "left")

                // Join with Merchant Features
                .join(merchFeats, txFeats.col("MERCHANT_ID").equalTo(merchFeats.col("MF_ID")), "left")

                // Join with Category Features
                .join(merchCatFeats, txFeats.col("CATEGORY").equalTo(merchCatFeats.col("MCF_CATEGORY")), "left")

                // Join with Customer-Merchant Interaction Features
                .join(custMerchFeats,
                        txFeats.col("CUSTOMER_ID").equalTo(custMerchFeats.col("CM_CUSTOMER_ID"))
                                .and(txFeats.col("MERCHANT_ID").equalTo(custMerchFeats.col("CM_MERCHANT_ID"))), "left")

                // Join with Customer-Category Interaction Features
                .join(custMerchCatFeats,
                        txFeats.col("CUSTOMER_ID").equalTo(custMerchCatFeats.col("CMC_CUSTOMER_ID"))
                                .and(txFeats.col("CATEGORY").equalTo(custMerchCatFeats.col("CMC_CATEGORY"))), "left")

                // Join with Fraud Labels (using transaction ID)
                .join(fraudTrans, txFeats.col("ID").equalTo(fraudTrans.col("TRANS_ID")), "left");

        // 3. Clean up join-key columns to avoid duplicates in the final CSV
        Dataset<Row> trainingRecords = combined.drop(
                "CF_ID", "MF_ID", "MCF_CATEGORY",
                "CM_CUSTOMER_ID", "CM_MERCHANT_ID",
                "CMC_CUSTOMER_ID", "CMC_CATEGORY", "TRANS_ID"
        );

        // 4. Persist Result
        // Note: coalesce(1) is removed for memory safety. Spark will write multiple files.
        trainingRecords.write()
                .mode("overwrite")
                .option("header", "true")
                .csv(outputDir);

        System.out.println("Combined training records generated in: " + outputDir);

        spark.stop();
    }
}