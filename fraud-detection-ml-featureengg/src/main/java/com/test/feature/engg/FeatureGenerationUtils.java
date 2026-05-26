package com.test.feature.engg;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import static org.apache.spark.sql.functions.*;

public class FeatureGenerationUtils {

    //CustomerHeader: ID,ACCOUNT_AGE_DAYS,AGE,CITY,CREDIT_SCORE,FIRST_NAME,
    // LAST_NAME
    //CustemerDeviceHeader: ID,DEVICE_TYPE,OS,STATUS,CUST_ID
    //MerchanHeader: ID,CATEGORY,CITY,NAME
    //TransactionHeader: ID,STATUS,TRANS_TIMESTAMP,VALUEUSD,CUSTOMER_ID,DEVICE_ID,MERCHANT_ID
    //FraudTransactionHeader: ID,IS_FRAUD,TRANS_ID

    public static Dataset<Row> generateTransactionFeatures(Dataset<Row> tx, Dataset<Row> cust, Dataset<Row> merch, Dataset<Row> fraud) {
        // Headers: Transaction(ID, TRANS_TIMESTAMP, VALIUEUSD, CUSTOMER_ID, MERCHANT_ID)
        // Headers: Customers(ID, CITY), Merchants(ID, CATEGORY, CITY), Fraud(TRANS_ID, IS_FRAUD)
        return tx.join(cust, tx.col("CUSTOMER_ID").equalTo(cust.col("ID")), "left")
                .join(merch, tx.col("MERCHANT_ID").equalTo(merch.col("ID")), "left")
                .join(fraud, tx.col("ID").equalTo(fraud.col("TRANS_ID")), "left")
                .withColumn("ts", to_timestamp(col("TRANS_TIMESTAMP")))
                .withColumn("TransactionHour", hour(col("ts")))
                .withColumn("TransactionDayOfWeek", date_format(col("ts"), "u"))
                .withColumn("TransactionIsWeekend", when(date_format(col("ts"), "u").geq(6), true).otherwise(false))
                .withColumn("InCityTransaction", cust.col("CITY").equalTo(merch.col("CITY")))
                .withColumn("FraudLabel", when(col("IS_FRAUD").equalTo("TRUE"), 1).otherwise(0))
                .select(tx.col("*"), merch.col("CATEGORY"), col("TransactionHour"),
                        col("TransactionDayOfWeek"), col("TransactionIsWeekend"),
                        col("InCityTransaction"), col("FraudLabel"));
    }

    public static Dataset<Row> generateCustomerFeatures(Dataset<Row> tx, Dataset<Row> cust, Dataset<Row> fraud, Dataset<Row> dev) {
        // Headers: Transaction(CUSTOMER_ID, VALIUEUSD), CustomerDevices(CUST_ID)
        Dataset<Row> txAggs = tx.groupBy("CUSTOMER_ID").agg(
                count("ID").as("CustomerTransactionCount"),
                sum("VALIUEUSD").as("CustomerTotalAmount"),
                avg("VALIUEUSD").as("CustomerAvgAmount"),
                max("VALIUEUSD").as("HighestTransactionValue"),
                min("VALIUEUSD").as("LowestTransactionValue")
        );

        Dataset<Row> fraudAgg = tx.join(fraud, tx.col("ID").equalTo(fraud.col("TRANS_ID")))
                .filter(col("IS_FRAUD").equalTo("TRUE"))
                .groupBy("CUSTOMER_ID").agg(count("*").as("CustomerFraudCount"));

        Dataset<Row> devAgg = dev.groupBy("CUST_ID").agg(countDistinct("ID").as("CustomerDeviceCount"));

        return cust.join(txAggs, cust.col("ID").equalTo(txAggs.col("CUSTOMER_ID")), "left")
                .join(fraudAgg, cust.col("ID").equalTo(fraudAgg.col("CUSTOMER_ID")), "left")
                .join(devAgg, cust.col("ID").equalTo(devAgg.col("CUST_ID")), "left")
                .withColumn("CF_ID", cust.col("ID"))
                .select(col("CF_ID"), col("ACCOUNT_AGE_DAYS"), col("AGE").as("CustomerAge"),
                        col("CREDIT_SCORE").as("CustomerCreditScore"), col("CustomerTransactionCount"),
                        col("CustomerTotalAmount"), col("CustomerAvgAmount"), col("CustomerFraudCount"),
                        col("CustomerDeviceCount"));
    }

    public static Dataset<Row> generateMerchantFeatures(Dataset<Row> tx, Dataset<Row> merch, Dataset<Row> fraud) {
        Dataset<Row> txFraud = tx.join(fraud, tx.col("ID").equalTo(fraud.col("TRANS_ID")), "left");
        return txFraud.groupBy("MERCHANT_ID").agg(
                count("ID").as("MerchantTransactionCount"),
                sum("VALIUEUSD").as("MerchantTotalAmount"),
                avg("VALIUEUSD").as("MerchantAvgAmount"),
                sum(when(col("IS_FRAUD").equalTo("TRUE"), 1).otherwise(0)).as("MerchantFraudCount")
        ).withColumn("MF_ID", col("MERCHANT_ID")).drop("MERCHANT_ID");
    }

    public static Dataset<Row> generateMerchantCategoryFeatures(Dataset<Row> tx, Dataset<Row> merch, Dataset<Row> fraud) {
        Dataset<Row> txFull = tx.join(merch, tx.col("MERCHANT_ID").equalTo(merch.col("ID")))
                .join(fraud, tx.col("ID").equalTo(fraud.col("TRANS_ID")), "left");
        return txFull.groupBy("CATEGORY").agg(
                count("ID").as("CategoryTransactionCount"),
                sum("VALIUEUSD").as("CategoryTotalAmount"),
                sum(when(col("IS_FRAUD").equalTo("TRUE"), 1).otherwise(0)).as("CategoryFraudCount")
        ).withColumn("MCF_CATEGORY", col("CATEGORY")).drop("CATEGORY");
    }

    public static Dataset<Row> generateCustomerMerchantFeatures(Dataset<Row> tx) {
        return tx.groupBy("CUSTOMER_ID", "MERCHANT_ID").agg(
                        count("ID").as("CustMerchTransactionCount"),
                        sum("VALIUEUSD").as("CustMerchTotalAmount")
                ).withColumn("CM_CUSTOMER_ID", col("CUSTOMER_ID")).withColumn("CM_MERCHANT_ID", col("MERCHANT_ID"))
                .drop("CUSTOMER_ID", "MERCHANT_ID");
    }

    public static Dataset<Row> generateCustomerMerchantCategoryFeatures(Dataset<Row> tx, Dataset<Row> merch) {
        Dataset<Row> txMerch = tx.join(merch, tx.col("MERCHANT_ID").equalTo(merch.col("ID")));
        return txMerch.groupBy("CUSTOMER_ID", "CATEGORY").agg(
                        count("ID").as("CustMerchCatTransactionCount"),
                        sum("VALIUEUSD").as("CustMerchCatTotalAmount")
                ).withColumn("CMC_CUSTOMER_ID", col("CUSTOMER_ID")).withColumn("CMC_CATEGORY", col("CATEGORY"))
                .drop("CUSTOMER_ID", "CATEGORY");
    }
}