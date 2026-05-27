package com.test.feature.engg;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

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
                .getOrCreate();

        // 1. Read Raw Datasets
        //CustomerHeader: ID,ACCOUNT_AGE_DAYS,AGE,CITY,CREDIT_SCORE,FIRST_NAME,
        // LAST_NAME
        System.out.println("Reading files and creating Spark datasets..");
        Dataset<Row> customers = spark.read().option("header", "true").csv(inputDir + "/customer*.csv");
        //CustemerDeviceHeader: ID,DEVICE_TYPE,OS,STATUS,CUST_ID
        Dataset<Row> customerDevices = spark.read().option("header", "true").csv(inputDir + "/customer_device*.csv");
        //MerchanHeader: ID,CATEGORY,CITY,NAME
        Dataset<Row> merchants = spark.read().option("header", "true").csv(inputDir + "/merchant*.csv");
        //TransactionHeader: ID,STATUS,TRANS_TIMESTAMP,VALUEUSD,CUSTOMER_ID,DEVICE_ID,MERCHANT_ID
        Dataset<Row> transactions = spark.read().option("header", "true")
                .option("spark.sql.legacy.timeParserPolicy", "LEGACY")
                .csv(inputDir + "/transaction*.csv");
        //FraudTransactionHeader: ID,IS_FRAUD,TRANS_ID
        Dataset<Row> fraudTransactions = spark.read().option("header", "true").csv(inputDir + "/fraud_transaction*.csv");

        System.out.println("Generating feature groups.. for transaction and " +
                "customers..");
        // 2. Generate Feature Groups
        Dataset<Row> transFeats = FeatureGenerationUtils.generateTransactionFeatures(transactions,
                customers, merchants, fraudTransactions);
        Dataset<Row> custFeats =
                FeatureGenerationUtils.generateCustomerFeatures(transactions, customers,  fraudTransactions, customerDevices);
        /*Dataset<Row> merchFeats =
                FeatureGenerationUtils.generateMerchantFeatures(transactions, merchants, fraudTransactions);
        Dataset<Row> merchCatFeats = FeatureGenerationUtils.generateMerchantCategoryFeatures(transactions, merchants, fraudTransactions);
        Dataset<Row> custMerchFeats = FeatureGenerationUtils.generateCustomerMerchantFeatures(transactions);
        Dataset<Row> custMerchCatFeats = FeatureGenerationUtils
        .generateCustomerMerchantCategoryFeatures(transactions, merchants);
         */

        System.out.println("Persisting transaction and customer feature " +
                "groups to files.. ");
        // 3. Persist Individual Groups
        transFeats.write().mode("overwrite").option("header", "true").csv(outputDir + "/TransactionFeatures");
        custFeats.write().mode("overwrite").option("header", "true").csv(outputDir + "/CustomerFeatures");
        /*
        merchFeats.write().mode("overwrite").option("header", "true").csv(outputDir + "/MerchantFeatures");
        merchCatFeats.write().mode("overwrite").option("header", "true").csv(outputDir + "/MerchantCategoryFeatures");
        custMerchFeats.write().mode("overwrite").option("header", "true").csv(outputDir + "/CustomerMerchantFeatures");
        custMerchCatFeats.write().mode("overwrite").option("header", "true")
        .csv(outputDir + "/CustomerMerchantCategoryFeatures");
         */

        // 4. Combine for Training Records
        /*Dataset<Row> trainingRecords = transFeats
                .join(custFeats, transFeats.col("CUSTOMER_ID").equalTo(custFeats.col("CF_ID")), "left")
                .join(merchFeats, transFeats.col("MERCHANT_ID").equalTo(merchFeats.col("MF_ID")), "left")
                .join(merchCatFeats, transFeats.col("CATEGORY").equalTo(merchCatFeats.col("MCF_CATEGORY")), "left")
                .join(custMerchFeats,
                        transFeats.col("CUSTOMER_ID").equalTo(custMerchFeats.col("CM_CUSTOMER_ID"))
                                .and(transFeats.col("MERCHANT_ID").equalTo(custMerchFeats.col("CM_MERCHANT_ID"))), "left")
                .join(custMerchCatFeats,
                        transFeats.col("CUSTOMER_ID").equalTo(custMerchCatFeats.col("CMC_CUSTOMER_ID"))
                                .and(transFeats.col("CATEGORY").equalTo(custMerchCatFeats.col("CMC_CATEGORY"))), "left");

        trainingRecords.write().mode("overwrite").option("header", "true")
        .csv(outputDir + "/TrainingRecords");
         */

        spark.stop();
    }
}