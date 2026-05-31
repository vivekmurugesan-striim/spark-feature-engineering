package com.test.feature.engg;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import static com.test.feature.engg.FeatureGenerationUtils.filterTxRecords;

public class TransactionRecordFilter {
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

        System.out.println("Count of filtered records::" + transactions.count());

        transactions.write().mode("overwrite").option("header", "true").csv(outputDir +
                "/" + "Transactions");

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