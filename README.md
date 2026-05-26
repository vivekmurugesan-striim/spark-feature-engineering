# spark-feature-engineering
Spark data processing that generates features for the Fraud detection ML training

# To run
spark-submit --master local[4] --class "com.test.feature.engg.SparkDriver"  target/fraud-detection-ml-featureengg-1.0-SNAPSHOT-jar-with-dependencies.jar /home/ec2-user/ml-data/data output >target/out.txt 2>target/err.txt &
