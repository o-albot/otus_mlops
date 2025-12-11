## Бакет с очищенными данными в формате parquet
s3://oa-otus-mlops/  
## Проверка и очистка данных
В папке ноутбук для анализа датасета и скрипт для очиски данных.  
Скрипт загружал на мастерноду в hdfs и запускал командой:  

spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --executor-memory 4g \
  --num-executors 10 \
  hdfs:///user/ubuntu/data/clean_fraud_s3.py \
  --input s3a://otus-mlops-source-data/ \
  --output s3a://oa-otus-mlops/clean_data/
