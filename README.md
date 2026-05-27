# spark-feature-engineering
Spark data processing that generates features for the Fraud detection ML training

# To run
spark-submit --master local[*] --driver-memory 4g --executor-memory 4g --class "com.test.feature.engg.SparkDriver"  target/fraud-detection-ml-featureengg-1.0-SNAPSHOT-jar-with-dependencies.jar /home/ec2-user/ml-data/data output >target/out.txt 2>target/err.txt &

# To run training record generator
spark-submit --master local[*] --driver-memory 4g --executor-memory 4g --class "com.test.feature.engg.TrainingRecordGenerator"  target/fraud-detection-ml-featureengg-1.0-SNAPSHOT-jar-with-dependencies.jar /home/ec2-user/ml-data/features output >target/out.txt 2>target/err.txt &

# To trigger model training
spark-submit --master local[*] --driver-memory 4g --executor-memory 4g --class "com.test.ml.training.FraudDetectionMLTraining"  target/fraud-detection-ml-featureengg-1.0-SNAPSHOT-jar-with-dependencies.jar /home/ec2-user/ml-data/training-records output >target/out.txt 2>target/err.txt &
