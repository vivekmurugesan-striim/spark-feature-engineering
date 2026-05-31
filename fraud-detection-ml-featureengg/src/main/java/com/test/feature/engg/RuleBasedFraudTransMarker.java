package com.test.feature.engg;

import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.expressions.Window;

import java.util.List;

import static com.test.feature.engg.FeatureGenerationUtils.filterTxRecords;
import static org.apache.spark.sql.functions.*;
import static org.apache.spark.sql.functions.col;

public class RuleBasedFraudTransMarker {

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
                spark.read().option("header", "true")
                        .option("inferSchema",
            "true").csv(inputDir + "/customer.*.csv");
        Dataset<Row> merchants =
                spark.read().option("header", "true")
                .option("inferSchema", "true").csv(inputDir + "/merchant*.csv");
        Dataset<Row> transactions =
                filterTxRecords(spark.read().option("header", "true").option(
                        "inferSchema", "true").csv(inputDir + "/transaction*" +
                        ".csv"));

        // 2. Generate Features

        Dataset<Row> txFeatures = generateTransactionFeatures(transactions,
                customers, merchants);
        Dataset<Row> enrichedTx = enrichTx(transactions);
        Dataset<Row> customerFeatures = generateCustomerFeatures(enrichedTx,
                customers);
        Dataset<Row> merchantFeatures = generateMerchantFeatures(enrichedTx,
                merchants);

        Dataset<Row> fraudFlaggedDataset =
                generateFraudTrans(txFeatures,
                customerFeatures, merchantFeatures);

        System.out.println("Generated:: fraud flagged dataset" +
                " with count::" + fraudFlaggedDataset.count());
        fraudFlaggedDataset.printSchema();

        // 3. Persist result
        customerFeatures.write().mode("overwrite").option("header", "true").csv(outputDir +
                "/" + "FraudFlaggedDataset");

        spark.close();

    }

    public static Dataset<Row> generateFraudTrans(Dataset<Row> txFeatures,
                                                  Dataset<Row> custFeatrues,
                                                  Dataset<Row> merchFeatures) {
        Dataset<Row> joined = txFeatures
                .join(custFeatrues,
                txFeatures.col("CUSTOMER_ID").equalTo(custFeatrues.col("C_ID")),
                "inner")
                .join(custFeatrues,
                        txFeatures.col("MERCHANT_ID").equalTo(merchFeatures.col(
                                "M_ID")),
                        "inner");

        // --- 3. Apply Fraud Rules to create the IS_FRAUD flag ---

        // Rule 1: VALUE_USD >= CustomerAvgAmount + CustomerSDAmount
        Column rule1 = col("VALUE_USD").geq(
                col("CustomerAvgAmount").plus(col("CustomerSDAmount")));

        // Rule 2: VALUE_USD >= HighestTransactionValue - CustomerSDAmount
        Column rule2 = col("VALUE_USD").geq(
                col("HighestTransactionValue").minus(col("CustomerSDAmount")));

        // Rule 3: InCityTransaction != true (Assuming 0 for false/not in city)
        Column rule3 = col("InCityTransaction").equalTo(lit(0));

        // Rule 4: (TransactionHour >= 23 OR TransactionHour <= 7) AND (TransactionIsWeekend != true)
        // This handles late night/early morning hours (23, 0, 1, 2, 3, 4, 5, 6, 7) on weekdays.
        Column rule4 = col("TransactionHour").geq(lit(23))
                .or(col("TransactionHour").leq(lit(7)))
                .and(col("TransactionIsWeekend").equalTo(lit(0)));

        // Rule 5: (TransactionHour >= 1 AND TransactionHour <= 9) AND (TransactionIsWeekend == true)
        // This handles early morning hours (1, 2, ..., 9) on weekends.
        Column rule5 = col("TransactionHour").geq(lit(1))
                .and(col("TransactionHour").leq(lit(9)))
                .and(col("TransactionIsWeekend").equalTo(lit(1)));

        // Rule 6: VALUE_USD >= MerchantAvgAmount + MerchantSDAmount
        Column rule6 = col("VALUE_USD").geq(
                col("MerchantAvgAmount").plus(col("MerchantSDAmount")));

        // The transaction is fraudulent if ANY of the rules are true (OR logic)
        Column combinedFraudCondition = rule1.or(rule2).or(rule3).or(rule4).or(rule5).or(rule6);


        // --- 4. Select Final Dataset ---
        // Create the final Dataset<Row> with TRANS_ID and IS_FRAUD
        Dataset<Row> fraudFlagsDataset = joined
                .withColumn("IS_FRAUD",
                        when(combinedFraudCondition, lit(1)).otherwise(lit(0)))
                .select(
                        col("TX.ID").alias("TRANS_ID"),
                        col("IS_FRAUD")
                );

        long nonFraudCount =
                fraudFlagsDataset.filter(col("IS_FRAUD").equalTo(0)).count();

        long fraudCount =
                fraudFlagsDataset.filter(col("IS_FRAUD").equalTo(1)).count();

        System.out.println("Non_fraud Count : " + nonFraudCount);
        System.out.println("Fraud count::" + fraudCount);
        System.out.println("Total count::" + fraudFlagsDataset.count());

        return fraudFlagsDataset;

    }

    /**
     * Generates the following features,
     *  - C_ID
     *  - CustomerAvgAmount
     *  - CustomerSDAmount
     *  - HighestTransactionValue
     *
     * @param tx
     * @param cust
     * @return
     */
    public static Dataset<Row> generateCustomerFeatures(Dataset<Row> tx,
                                                        Dataset<Row> cust) {
        Dataset<Row> base = tx.groupBy("CUSTOMER_ID").agg(
                avg("amount").as("CustomerAvgAmount"),
                stddev("amount").as("CustomerSDAmount"),
                max("amount").as("HighestTransactionValue"));

        return cust.withColumn("C_ID", cust.col("ID"))
                .join(base, cust.col("ID").equalTo(base.col("CUSTOMER_ID")), "left")
                .drop("CUSTOMER_ID");
    }

    /**
     * Generating features,
     *  - M_ID
     *  - MerchantAvgAmount
     *  - MerchantSDAmount
     * @param t
     * @param merch
     * @return
     */
    public static Dataset<Row> generateMerchantFeatures(Dataset<Row> t, Dataset<Row> merch) {

        // Aggregates - Using tx.col("ID") to be explicit
        Dataset<Row> base = t.groupBy("MERCHANT_ID").agg(
                avg("amount").as("MerchantAvgAmount"),
                stddev("amount").as("MerchantSDAmount"));


        return merch.withColumn("M_ID", merch.col("ID"))
                .join(base, merch.col("ID").equalTo(base.col("MERCHANT_ID")),
                        "left")
                .drop("MERCHANT_ID");
    }

    // Helper: Enrich Transactions with time components and fraud status
    private static Dataset<Row> enrichTx(Dataset<Row> tx) {

        return tx.withColumn("ts", to_timestamp(col("TRANS_TIMESTAMP")))
                .withColumn("amount", col("VALUEUSD"))
                .withColumn("date", to_date(col("ts")))
                .withColumn("hour", hour(col("ts")))
                .withColumn("day", dayofmonth(col("ts")))
                .withColumn("week", weekofyear(col("ts")))
                .withColumn("month", month(col("ts")))
                .withColumn("year", year(col("ts")))
                .withColumn("dow", date_format(col("ts"), "u"));
    }

    /**
     * Generates the following features,
     *  - ID
     *  - STATUS
     *  - TRANS_TIMESTAMP
     *  - VALUE_USD
     *  - CUSTOMER_ID
     *  - DEVICE_ID
     *  - MERCHANT_ID
     *  - TransactionHour
     *  - TransactionDayOfWeek
     *  - TransactionIsWeekend
     *  - TransactionMonth
     *  - InCityTransaction
     *  - CATEGORY
     * @param tx
     * @param cust
     * @param merch
     * @return
     */
    public static Dataset<Row> generateTransactionFeatures(Dataset<Row> tx,
                                                           Dataset<Row> cust, Dataset<Row> merch) {

        cust = cust.select(cust.col("ID"), cust.col("CITY"));
        merch = merch.select(merch.col("ID"), merch.col("CITY"), merch.col(
                "CATEGORY"));

        Dataset<Row> joined = tx.join(merch, tx.col("MERCHANT_ID").equalTo(merch.col("ID")),
                "inner");

        joined = joined
                .join(cust, tx.col("CUSTOMER_ID").equalTo(cust.col("ID")),
                        "inner");

        Dataset<Row> result = joined
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

        return result;
    }

}
