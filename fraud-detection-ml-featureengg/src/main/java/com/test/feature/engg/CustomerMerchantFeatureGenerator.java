package com.test.feature.engg;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.expressions.WindowSpec;

import static com.test.feature.engg.FeatureGenerationUtils.enrichTx;
import static org.apache.spark.sql.functions.*;
import static org.apache.spark.sql.functions.avg;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.count;
import static org.apache.spark.sql.functions.max;
import static org.apache.spark.sql.functions.sum;

public class CustomerMerchantFeatureGenerator {
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
        Dataset<Row> transactions = spark.read().option("header", "true").option("inferSchema", "true").csv(inputDir + "/transaction*.csv");
        Dataset<Row> fraud = spark.read().option("header", "true").option("inferSchema", "true").csv(inputDir + "/fraud_transaction*.csv");

        // 2. Generate Features
        Dataset<Row> merchantFeatures = generateCustomerMerchantFeatures(transactions,
                fraud);

        // 3. Persist result
        merchantFeatures.write().mode("overwrite").option("header", "true").csv(outputDir +
                "/" + "MerchantFeatures");

        spark.close();
    }

    public static Dataset<Row> generateCustomerMerchantFeatures(Dataset<Row> tx, Dataset<Row> fraud) {
        Dataset<Row> t = enrichTx(tx, fraud).withColumn("unix_ts", unix_timestamp(col("ts")));

        WindowSpec window = Window.partitionBy("CUSTOMER_ID", "MERCHANT_ID").orderBy("ts");
        Dataset<Row> timeDelta = t.withColumn("prev_ts", lag("unix_ts", 1).over(window))
                .groupBy("CUSTOMER_ID", "MERCHANT_ID").agg(avg(col("unix_ts").minus(col("prev_ts"))).as("TimeSinceLastTx"));

        Dataset<Row> aggs = t.groupBy("CUSTOMER_ID", "MERCHANT_ID").agg(
                count(tx.col("ID")).as("CustMerchTransactionCount"),
                sum("amount").as("CustMerchTotalAmount"),
                avg("amount").as("CustMerchAvgAmount"),
                sum("is_fraud_num").as("CustMerchFraudCount"),
                max("amount").as("HighestTransactionValueForMerchantByCustomer")
        ).withColumn("CustMerchFraudRate", col("CustMerchFraudCount").divide(col("CustMerchTransactionCount")));

        return aggs.join(timeDelta, new String[]{"CUSTOMER_ID", "MERCHANT_ID"}, "left")
                .withColumn("CM_CUSTOMER_ID", col("CUSTOMER_ID")).withColumn("CM_MERCHANT_ID", col("MERCHANT_ID")).drop("CUSTOMER_ID", "MERCHANT_ID");
    }
}
