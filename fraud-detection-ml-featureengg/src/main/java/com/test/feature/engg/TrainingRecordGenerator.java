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
        Dataset<Row> combined = txFeats.as("tx")
                // Join Customer Features
                .join(custFeats.as("cust"), col("tx.CUSTOMER_ID").equalTo(col("cust.C_ID")), "left")
                .drop(col("cust.C_ID"))

                // Join Merchant Features (CATEGORY conflict now resolved by the drop above)
                .join(merchFeats.as("merch"), col("tx.MERCHANT_ID").equalTo(col("merch.M_ID")), "left")
                .drop(col("merch.M_ID")).drop("merch.CATEGORY")

                // Join Merchant Category Features
                .join(merchCatFeats.as("mcf"), col("tx.CATEGORY").equalTo(col("mcf.MCF_CATEGORY")), "left")
                .drop(col("mcf.MCF_CATEGORY"))

                // Join Customer-Merchant Interaction
                .join(custMerchFeats.as("cm"),
                        col("tx.CUSTOMER_ID").equalTo(col("cm.CM_CUSTOMER_ID"))
                                .and(col("tx.MERCHANT_ID").equalTo(col("cm.CM_MERCHANT_ID"))), "left")
                .drop(col("cm.CM_CUSTOMER_ID")).drop(col("cm.CM_MERCHANT_ID"))

                // Join Customer-Category Interaction
                .join(custMerchCatFeats.as("cmc"),
                        col("tx.CUSTOMER_ID").equalTo(col("cmc.CMC_CUSTOMER_ID"))
                                .and(col("tx.CATEGORY").equalTo(col("cmc.CMC_CATEGORY"))), "left")
                .drop(col("cmc.CMC_CUSTOMER_ID")).drop(col("cmc.CMC_CATEGORY"))

                // Join Fraud Labels
                .join(fraudTrans.as("f"), col("tx.ID").equalTo(col("f.TRANS_ID")), "left")
                .drop(col("f.TRANS_ID"));

        // 3. Clean up join-key columns to avoid duplicates in the final CSV
        /*Dataset<Row> trainingRecords = combined.drop(
                "CF_ID", "MF_ID", "MCF_CATEGORY",
                "CM_CUSTOMER_ID", "CM_MERCHANT_ID",
                "CMC_CUSTOMER_ID", "CMC_CATEGORY", "TRANS_ID"
        );*/

        Dataset<Row> trainingRecords = combined.drop("cust.ID", "f.ID");

        System.out.println("TrainRecord structure::");
        System.out.println(trainingRecords.describe().toString());

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