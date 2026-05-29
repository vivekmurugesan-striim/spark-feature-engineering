package com.test.feature.engg;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Arrays;
import java.util.List;

import static com.test.feature.engg.FeatureGenerationUtils.filterTxRecords;

public class SparkDriver {
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
        Dataset<Row> customers =
                spark.read().option("header", "true").option("inferSchema", "true").csv(inputDir + "/customer.*.csv");
        Dataset<Row> devices = spark.read().option("header", "true").option("inferSchema", "true").csv(inputDir + "/customer_device*.csv");
        Dataset<Row> merchants = spark.read().option("header", "true").option("inferSchema", "true").csv(inputDir + "/merchant*.csv");
        Dataset<Row> transactions = filterTxRecords(spark.read().option(
                "header", "true").option("inferSchema", "true").csv(inputDir + "/transaction*.csv"));
        Dataset<Row> fraud = spark.read().option("header", "true").option("inferSchema", "true").csv(inputDir + "/fraud_transaction*.csv");

        // 2. Generate Feature Groups
        Dataset<Row> transFeats = FeatureGenerationUtils.generateTransactionFeatures(transactions, customers, merchants, fraud);
        Dataset<Row> merchFeats = FeatureGenerationUtils.generateMerchantFeatures(transactions, merchants, fraud);
        Dataset<Row> custFeats = FeatureGenerationUtils.generateCustomerFeatures(transactions, customers, fraud, devices, merchants);
        Dataset<Row> merchCatFeats = FeatureGenerationUtils.generateMerchantCategoryFeatures(transactions, merchants, fraud);
        Dataset<Row> custMerchFeats = FeatureGenerationUtils.generateCustomerMerchantFeatures(transactions, fraud);
        Dataset<Row> custMerchCatFeats = FeatureGenerationUtils.generateCustomerMerchantCategoryFeatures(transactions, merchants, fraud);

        // 3. Persist individual feature sets using a List instead of an Array
        List<Dataset<Row>> sets = Arrays.asList(transFeats, merchFeats, custFeats, merchCatFeats, custMerchFeats, custMerchCatFeats);
        String[] names = {"TransactionFeatures", "MerchantFeatures", "CustomerFeatures", "MerchantCategoryFeatures", "CustomerMerchantFeatures", "CustomerMerchantCategoryFeatures"};

        for (int i = 0; i < names.length; i++) {
            sets.get(i).write().mode("overwrite").option("header", "true").csv(outputDir + "/" + names[i]);
        }


        // 4. Combine for Training Records
        Dataset<Row> trainingRecords = transFeats
                .join(custFeats, transFeats.col("CUSTOMER_ID").equalTo(custFeats.col("C_ID")), "left")
                .join(merchFeats, transFeats.col("MERCHANT_ID").equalTo(merchFeats.col("M_ID")), "left")
                .join(merchCatFeats, transFeats.col("CATEGORY").equalTo(merchCatFeats.col("MCF_CATEGORY")), "left")
                .join(custMerchFeats,
                        transFeats.col("CUSTOMER_ID").equalTo(custMerchFeats.col("CM_CUSTOMER_ID"))
                                .and(transFeats.col("MERCHANT_ID").equalTo(custMerchFeats.col("CM_MERCHANT_ID"))), "left")
                .join(custMerchCatFeats,
                        transFeats.col("CUSTOMER_ID").equalTo(custMerchCatFeats.col("CMC_CUSTOMER_ID"))
                                .and(transFeats.col("CATEGORY").equalTo(custMerchCatFeats.col("CMC_CATEGORY"))), "left")
                .drop("C_ID", "M_ID", "MCF_CATEGORY", "CM_CUSTOMER_ID", "CM_MERCHANT_ID", "CMC_CUSTOMER_ID", "CMC_CATEGORY");

        trainingRecords.write().mode("overwrite").option("header", "true").csv(outputDir + "/TrainingRecords");

        spark.stop();
    }
}