package com.test.feature.engg;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static com.test.feature.engg.FeatureGenerationUtils.enrichTx;
import static org.apache.spark.sql.functions.*;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.max;
import static org.apache.spark.sql.functions.min;
import static org.apache.spark.sql.functions.sum;

public class MerchantCategoryFeatureGenerator {
    public static void main(String[] args){

        if (args.length < 2) {
            System.err.println("Usage: SparkDriver <input-dir> <output-dir>");
            System.exit(1);
        }

        String inputDir = args[0];
        String outputDir = args[1];

        SparkSession spark = SparkSession.builder()
                .appName("Banking Feature Engineering")
                .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
                .getOrCreate();

        // 1. Load Datasets
        Dataset<Row> merchants = spark.read().option("header", "true").option("inferSchema", "true").csv(inputDir + "/merchant*.csv");
        Dataset<Row> transactions = spark.read().option("header", "true").option("inferSchema", "true").csv(inputDir + "/transaction*.csv");
        Dataset<Row> fraud = spark.read().option("header", "true").option("inferSchema", "true").csv(inputDir + "/fraud_transaction*.csv");

        // 2. Generate Features
        Dataset<Row> merchantFeatures = generateMerchantCategoryFeatures(transactions,
                merchants,fraud);

        // 3. Persist result
        merchantFeatures.write().mode("overwrite").option("header", "true").csv(outputDir +
                "/" + "MerchantCategoryFeatures");

        spark.close();
    }

    public static Dataset<Row> generateMerchantCategoryFeatures(Dataset<Row> tx, Dataset<Row> merch, Dataset<Row> fraud) {
        Dataset<Row> t = enrichTx(tx, fraud).join(merch, tx.col("MERCHANT_ID").equalTo(merch.col("ID")));

        return t.groupBy("CATEGORY").agg(
                        count(tx.col("ID")).as("CategoryTransactionCount"),
                        sum("amount").as("CategoryTotalAmount"),
                        avg("amount").as("CategoryAvgAmount"),
                        sum("is_fraud_num").as("CategoryFraudCount"),
                        max("amount").as("HighestTransactionValueForCategory"),
                        min("amount").as("LowestTransactionValueForCategory")
                ).withColumn("CategoryFraudRate", col("CategoryFraudCount").divide(col("CategoryTransactionCount")))
                .withColumn("MCF_CATEGORY", col("CATEGORY")).drop("CATEGORY");
    }
}
