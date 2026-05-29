# spark-feature-engineering
Spark data processing that generates features for the Fraud detection ML training

# To run
spark-submit --master local[*] --driver-memory 4g --executor-memory 4g --class "com.test.feature.engg.SparkDriver"  target/fraud-detection-ml-featureengg-1.0-SNAPSHOT-jar-with-dependencies.jar /home/ec2-user/ml-data/data output >target/out.txt 2>target/err.txt &

#To Run transaction feature generation
spark-submit --master local[*] --driver-memory 4g --executor-memory 4g --class "com.test.feature.engg.TransactionFeatureGenerator"  target/fraud-detection-ml-featureengg-1.0-SNAPSHOT-jar-with-dependencies.jar /home/ec2-user/ml-data/data output >target/out.txt 2>target/err.txt &


# To run training record generator
spark-submit --master local[*] --driver-memory 4g --executor-memory 4g --class "com.test.feature.engg.TrainingRecordGenerator"  target/fraud-detection-ml-featureengg-1.0-SNAPSHOT-jar-with-dependencies.jar /home/ec2-user/ml-data/features output >target/out.txt 2>target/err.txt &

# To trigger model training
spark-submit --master local[*] --driver-memory 8g --executor-memory 8g --conf spark.memory.fraction=0.7 --conf spark.memory.storageFraction=0.3 --conf spark.network.timeout=1800s --conf spark.executor.heartbeatInterval=60s --conf spark.sql.shuffle.partitions=800 --conf spark.driver.maxResultSize=4g --conf spark.shuffle.service.enabled=true --conf spark.dynamicAllocation.enabled=true --conf spark.sql.broadcastTimeout=1800 --class "com.test.ml.training.FraudDetectionMLTraining"  target/fraud-detection-ml-featureengg-1.0-SNAPSHOT-jar-with-dependencies.jar /home/ec2-user/ml-data/training-records output >target/out.txt 2>target/err.txt &
