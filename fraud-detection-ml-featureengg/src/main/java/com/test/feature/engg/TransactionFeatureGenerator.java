package com.test.feature.engg;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.*;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.date_format;
import static org.apache.spark.sql.functions.month;
import static org.apache.spark.sql.functions.when;

public class TransactionFeatureGenerator {
    public static void main(String[] args) {
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
        Dataset<Row> customers = spark.read().option("header", "true").option("inferSchema", "true").csv(inputDir + "/customer*.csv");
        Dataset<Row> merchants = spark.read().option("header", "true").option("inferSchema", "true").csv(inputDir + "/merchant*.csv");

        // 2. Generate Features
        Dataset<Row> txFeatures = generateTransactionFeatures(transactions,
                customers, merchants, spark);

        // 3. Persist result
        txFeatures.write().mode("overwrite").option("header", "true").csv(outputDir +
                "/" + "TransactionFeatures");

        spark.close();

    }

    public static Dataset<Row> generateTransactionFeatures(Dataset<Row> tx,
                                                           Dataset<Row> cust, Dataset<Row> merch,
                                                           SparkSession spark) {

        cust = cust.select(cust.col("ID"), cust.col("CITY"));
        merch = merch.select(merch.col("ID"), merch.col("CITY"));

        Dataset<Row> joined = tx.join(cust,
                        tx.col("CUSTOMER_ID").equalTo(cust.col("ID")),
                        "left")
                .join(merch, tx.col("MERCHANT_ID").equalTo(merch.col("ID")),
                        "left");

        long txCount = tx.count();
        long joinedCount = joined.count();

        System.out.println("TxCount::" + txCount);
        System.out.println("JoinedCount::" + joinedCount);

        if(txCount != joinedCount){
            System.err.println("Transaction Count and joined count are not " +
                    "matchin hence exiting::" + txCount + " != " + joinedCount);
            spark.close();
            System.exit(1);
        }

        Dataset<Row> result = joined
                .withColumn("ts", to_timestamp(col("TRANS_TIMESTAMP")))
                .withColumn("TransactionHour", hour(col("ts")))
                .withColumn("TransactionDayOfWeek",
                        date_format(col("ts"), "u"))
                .withColumn("TransactionIsWeekend",
                        when(date_format(col("ts"), "u").geq(6), 1).otherwise(0))
                .withColumn("TransactionMonth", month(col("ts")))
                .withColumn("InCityTransaction", when(cust.col("CITY").equalTo(merch.col("CITY")), 1).otherwise(0))
                .select(tx.col("*"), merch.col("CATEGORY"), col("TransactionHour"),
                        col("TransactionDayOfWeek"), col("TransactionIsWeekend"), col("TransactionMonth"),
                        col("InCityTransaction"));


        long resultCount = result.count();

        System.out.println("TxCount::" + txCount);
        System.out.println("ResultCount::" + resultCount);

        if(txCount != resultCount){
            System.err.println("Transaction Count and joined count are not " +
                    "matchin hence exiting::" + txCount + " != " + joinedCount);
            spark.close();
            System.exit(1);
        }

        return result;

    }

}
