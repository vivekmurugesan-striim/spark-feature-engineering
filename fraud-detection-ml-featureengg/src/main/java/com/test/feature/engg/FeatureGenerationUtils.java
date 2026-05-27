package com.test.feature.engg;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.expressions.WindowSpec;

import static org.apache.spark.sql.functions.*;

public class FeatureGenerationUtils {

    //CustomerHeader: ID,ACCOUNT_AGE_DAYS,AGE,CITY,CREDIT_SCORE,FIRST_NAME,
    // LAST_NAME
    //CustemerDeviceHeader: ID,DEVICE_TYPE,OS,STATUS,CUST_ID
    //MerchanHeader: ID,CATEGORY,CITY,NAME
    //TransactionHeader: ID,STATUS,TRANS_TIMESTAMP,VALUEUSD,CUSTOMER_ID,DEVICE_ID,MERCHANT_ID
    //FraudTransactionHeader: ID,IS_FRAUD,TRANS_ID

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

    /**
     * Corrected Helper: Compute Mode (Top/Least Freq) for categorical dimensions.
     * Now preserves the idCol to allow for subsequent joins.
     */
    private static Dataset<Row> getMode(Dataset<Row> t, String idCol, String dimCol, String prefix) {
        Dataset<Row> counts = t.groupBy(idCol, dimCol).agg(count("*").as("c"));

        Dataset<Row> top = counts.withColumn("rn", row_number().over(Window.partitionBy(idCol).orderBy(desc("c"))))
                .filter(col("rn").equalTo(1))
                .select(col(idCol), col(dimCol).as("TopFreq" + prefix + dimCol));

        Dataset<Row> least = counts.withColumn("rn", row_number().over(Window.partitionBy(idCol).orderBy(asc("c"))))
                .filter(col("rn").equalTo(1))
                .select(col(idCol), col(dimCol).as("LeastFreq" + prefix + dimCol));

        // Use a simple join on the idCol string to avoid duplicate columns and keep the ID
        return top.join(least, idCol);
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

        Dataset<Row> modes = getMode(t, "MERCHANT_ID", "day", "Merchant").join(getMode(t, "MERCHANT_ID", "hour", "Merchant"), "MERCHANT_ID");

        return merch.withColumn("M_ID", merch.col("ID"))
                .join(base, merch.col("ID").equalTo(base.col("MERCHANT_ID")), "left")
                .join(daily, "MERCHANT_ID", "left").join(weekly, "MERCHANT_ID", "left").join(monthly, "MERCHANT_ID", "left")
                .join(modes, "MERCHANT_ID", "left").drop("MERCHANT_ID");
    }

    public static Dataset<Row> generateCustomerFeatures(Dataset<Row> tx, Dataset<Row> cust, Dataset<Row> fraud, Dataset<Row> dev, Dataset<Row> merch) {
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

    public static Dataset<Row> generateCustomerMerchantCategoryFeatures(Dataset<Row> tx, Dataset<Row> merch, Dataset<Row> fraud) {
        Dataset<Row> t = enrichTx(tx, fraud).join(merch, tx.col("MERCHANT_ID").equalTo(merch.col("ID")));

        return t.groupBy("CUSTOMER_ID", "CATEGORY").agg(
                        count(tx.col("ID")).as("CustMerchCatTransactionCount"),
                        sum("amount").as("CustMerchCatTotalAmount"),
                        avg("amount").as("CustMerchCatAvgAmount"),
                        sum("is_fraud_num").as("CustMerchCatFraudCount")
                ).withColumn("CustMerchCatFraudRate", col("CustMerchCatFraudCount").divide(col("CustMerchCatTransactionCount")))
                .withColumn("CMC_CUSTOMER_ID", col("CUSTOMER_ID")).withColumn("CMC_CATEGORY", col("CATEGORY")).drop("CUSTOMER_ID", "CATEGORY");
    }

    public static Dataset<Row> generateTransactionFeatures(Dataset<Row> tx, Dataset<Row> cust, Dataset<Row> merch, Dataset<Row> fraud) {
        return tx.join(cust, tx.col("CUSTOMER_ID").equalTo(cust.col("ID")), "left")
                .join(merch, tx.col("MERCHANT_ID").equalTo(merch.col("ID")), "left")
                .withColumn("ts", to_timestamp(col("TRANS_TIMESTAMP")))
                .withColumn("TransactionHour", hour(col("ts")))
                .withColumn("TransactionDayOfWeek", date_format(col("ts"), "u"))
                .withColumn("TransactionIsWeekend", when(date_format(col("ts"), "u").geq(6), 1).otherwise(0))
                .withColumn("TransactionMonth", month(col("ts")))
                .withColumn("InCityTransaction", when(cust.col("CITY").equalTo(merch.col("CITY")), 1).otherwise(0))
                .select(tx.col("*"), merch.col("CATEGORY"), col("TransactionHour"), col("TransactionDayOfWeek"), col("TransactionIsWeekend"), col("TransactionMonth"), col("InCityTransaction"));
    }
}