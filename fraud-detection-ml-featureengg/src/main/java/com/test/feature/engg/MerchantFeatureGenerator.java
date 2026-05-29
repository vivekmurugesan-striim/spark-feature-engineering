package com.test.feature.engg;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static com.test.feature.engg.FeatureGenerationUtils.enrichTx;
import static com.test.feature.engg.FeatureGenerationUtils.getMode;
import static org.apache.spark.sql.functions.*;
import static org.apache.spark.sql.functions.avg;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.max;
import static org.apache.spark.sql.functions.min;
import static org.apache.spark.sql.functions.sum;

public class MerchantFeatureGenerator {

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
        Dataset<Row> merchantFeatures = generateMerchantFeatures(transactions,
                merchants, fraud);

        // 3. Persist result
        merchantFeatures.write().mode("overwrite").option("header", "true").csv(outputDir +
                "/" + "MerchantFeatures");

        spark.close();
    }

    public static Dataset<Row> generateMerchantFeatures(Dataset<Row> tx, Dataset<Row> merch, Dataset<Row> fraud) {
        Dataset<Row> t = enrichTx(tx, fraud);

        // Aggregates - Using tx.col("ID") to be explicit
        Dataset<Row> base = t.groupBy("MERCHANT_ID").agg(
                count(tx.col("ID")).as("MerchantTransactionCount"),
                sum("amount").as("MerchantTotalAmount"),
                avg("amount").as("MerchantAvgAmount"),
                sum("is_fraud_num").as("MerchantFraudCount"),
                max("amount").as("HighestTransactionValueForMerchant"),
                min("amount").as("LowestTransactionValueForMerchant")
        ).withColumn("MerchantFraudRate", col("MerchantFraudCount").divide(col("MerchantTransactionCount")));

        Dataset<Row> daily = t.groupBy("MERCHANT_ID", "date").agg(sum("amount").as("s")).groupBy("MERCHANT_ID").agg(avg("s").as("DailyTransValueAvgForMerchant"));
        Dataset<Row> weekly = t.groupBy("MERCHANT_ID", "year", "week").agg(sum("amount").as("s")).groupBy("MERCHANT_ID").agg(avg("s").as("WeeklyTransValueAvgForMerchant"));
        Dataset<Row> monthly = t.groupBy("MERCHANT_ID", "year", "month").agg(sum("amount").as("s")).groupBy("MERCHANT_ID").agg(avg("s").as("MonthlyTransValueAvgForMerchant"));

        Dataset<Row> modes = getMode(t, "MERCHANT_ID", "day", "Merchant")
                .join(getMode(t, "MERCHANT_ID", "hour", "Merchant"), "MERCHANT_ID");

        return merch.withColumn("M_ID", merch.col("ID"))
                .join(base, merch.col("ID").equalTo(base.col("MERCHANT_ID")), "left")
                .join(daily, "MERCHANT_ID", "left").join(weekly, "MERCHANT_ID", "left").join(monthly, "MERCHANT_ID", "left")
                .join(modes, "MERCHANT_ID", "left").drop("MERCHANT_ID");
    }
}
