package com.test.feature.engg;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.expressions.Window;

import static com.test.feature.engg.FeatureGenerationUtils.filterTxRecords;
import static org.apache.spark.sql.functions.*;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.count;
import static org.apache.spark.sql.functions.countDistinct;
import static org.apache.spark.sql.functions.desc;
import static org.apache.spark.sql.functions.max;
import static org.apache.spark.sql.functions.min;
import static org.apache.spark.sql.functions.row_number;
import static org.apache.spark.sql.functions.sum;

public class CustomerFeatureGenerator {

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
        Dataset<Row> customers =
                spark.read().option("header", "true").option("inferSchema", "true").csv(inputDir + "/customer.*.csv");
        Dataset<Row> devices = spark.read().option("header", "true").option("inferSchema", "true").csv(inputDir + "/customer_device*.csv");
        Dataset<Row> merchants = spark.read().option("header", "true").option("inferSchema", "true").csv(inputDir + "/merchant*.csv");
        Dataset<Row> transactions =
                filterTxRecords(spark.read().option("header", "true").option(
                        "inferSchema", "true").csv(inputDir + "/transaction*" +
                        ".csv"));
        Dataset<Row> fraud = spark.read().option("header", "true").option("inferSchema", "true").csv(inputDir + "/fraud_transaction*.csv");


        // 2. Generate Features
        Dataset<Row> customerFeatures = generateCustomerFeatures(transactions,
                customers, fraud, devices, merchants);

        System.out.println("Generated:: customer features with count::" + customerFeatures.count());

        // 3. Persist result
        customerFeatures.write().mode("overwrite").option("header", "true").csv(outputDir +
                "/" + "CustomerFeatures");

        spark.close();

    }

    public static Dataset<Row> generateCustomerFeatures(Dataset<Row> tx, Dataset<Row> cust,
                                                        Dataset<Row> fraud, Dataset<Row> dev, Dataset<Row> merch) {
        Dataset<Row> t = enrichTx(tx, fraud).join(merch, tx.col("MERCHANT_ID").equalTo(merch.col("ID")));

        Dataset<Row> base = t.groupBy("CUSTOMER_ID").agg(
                count(tx.col("ID")).as("CustomerTransactionCount"),
                sum("amount").as("CustomerTotalAmount"),
                avg("amount").as("CustomerAvgAmount"),
                sum("is_fraud_num").as("CustomerFraudCount"),
                max("amount").as("HighestTransactionValue"),
                min("amount").as("LowestTransactionValue")
        ).withColumn("CustomerFraudRate", col("CustomerFraudCount").divide(col("CustomerTransactionCount")));

        Dataset<Row> devAgg = dev.groupBy("CUST_ID").agg(countDistinct(dev.col("ID")).as("CustomerDeviceCount"));

        Dataset<Row> catMode = t.groupBy("CUSTOMER_ID", "CATEGORY").agg(count("*").as("c"))
                .withColumn("rn", row_number().over(Window.partitionBy("CUSTOMER_ID").orderBy(desc("c"))))
                .filter(col("rn").equalTo(1)).select(col("CUSTOMER_ID").as("c_tmp"), col("CATEGORY").as("TopSpentCategory"));

        return cust.withColumn("C_ID", cust.col("ID"))
                .join(base, cust.col("ID").equalTo(base.col("CUSTOMER_ID")), "left")
                .join(devAgg, cust.col("ID").equalTo(devAgg.col("CUST_ID")), "left")
                .join(catMode, cust.col("ID").equalTo(catMode.col("c_tmp")), "left").drop("CUSTOMER_ID", "CUST_ID", "c_tmp");
    }

    // Helper: Enrich Transactions with time components and fraud status
    private static Dataset<Row> enrichTx(Dataset<Row> tx, Dataset<Row> fraud) {
        // Rename fraud.ID to avoid ambiguity with tx.ID
        Dataset<Row> renamedFraud = fraud.withColumnRenamed("ID", "FRAUD_PK_ID");

        return tx.join(renamedFraud, tx.col("ID").equalTo(renamedFraud.col("TRANS_ID")), "left")
                .withColumn("ts", to_timestamp(col("TRANS_TIMESTAMP")))
                .withColumn("amount", col("VALUEUSD"))
                .withColumn("date", to_date(col("ts")))
                .withColumn("hour", hour(col("ts")))
                .withColumn("day", dayofmonth(col("ts")))
                .withColumn("week", weekofyear(col("ts")))
                .withColumn("month", month(col("ts")))
                .withColumn("year", year(col("ts")))
                .withColumn("dow", date_format(col("ts"), "u"))
                .withColumn("is_fraud_num", when(col("IS_FRAUD").equalTo("TRUE"), 1).otherwise(0));
    }
}
