package com.test.feature.engg;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import static com.test.feature.engg.FeatureGenerationUtils.filterTxRecords;

public class RandomFradulentDataFilter {

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
        Dataset<Row> transactions =
                filterTxRecords(spark.read().option("header", "true")
                        .schema(buildTransRawDataSchema())
                        .csv(inputDir + "/transaction*.csv"));
        Dataset<Row> fraudTrans = spark.read().option("header", "true")
                .schema(buildFraudTransSchema())
                .csv(inputDir + "/fraud_transaction*.csv");

        System.out.println("Count of transaction records::"
                + transactions.count());
        System.out.println("Count of fraudtrans records::"
                + fraudTrans.count());

        // Rename fraud.ID to avoid ambiguity with tx.ID
        Dataset<Row> renamedFraud = fraudTrans.withColumnRenamed("ID",
                "FRAUD_PK_ID");

        Dataset<Row> filteredFraudTrans =
                transactions.join(renamedFraud,
                transactions.col("ID").equalTo(renamedFraud.col("TRANS_ID")),
                "inner")
                        .select(renamedFraud.col("TRANS_ID"),
                                renamedFraud.col("IS_FRAUD"));

        System.out.println("Count of filtered fraudTrans records::"
                + filteredFraudTrans.count());

        filteredFraudTrans.write().mode("overwrite").option("header", "true").csv(outputDir +
                "/" + "FraudTransactions");

    }

    private static StructType buildFraudTransSchema() {
        // 1. Explicit Schema Definition (CRITICAL: Removes double-pass inferSchema overhead)
        StructType schema = new StructType(new StructField[]{
                new StructField("ID", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("IS_FRAUD", DataTypes.IntegerType, true,
                        Metadata.empty()),
                new StructField("TRANS_ID", DataTypes.IntegerType,
                        true, Metadata.empty())});
        return schema;
    }

    private static StructType buildTransRawDataSchema() {
        // 1. Explicit Schema Definition (CRITICAL: Removes double-pass inferSchema overhead)
        StructType schema = new StructType(new StructField[]{
                new StructField("ID", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("STATUS", DataTypes.StringType, true, Metadata.empty()),
                new StructField("TRANS_TIMESTAMP", DataTypes.TimestampType, true, Metadata.empty()),
                new StructField("VALUEUSD", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("CUSTOMER_ID", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("DEVICE_ID", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("MERCHANT_ID", DataTypes.IntegerType,
                        true, Metadata.empty())});
        return schema;
    }

}
