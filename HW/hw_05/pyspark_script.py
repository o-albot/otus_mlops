import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, LongType, StringType, DoubleType, IntegerType
)
from pyspark.sql.functions import col, to_timestamp
import argparse


def main():
    parser = argparse.ArgumentParser(description="Очистка мошеннических транзакций из S3 → Parquet в S3")
    parser.add_argument("--input", required=True,
                        help="Путь к входным файлам в S3 (например, s3a://my-bucket/raw/transactions_*.txt)")
    parser.add_argument("--output", required=True,
                        help="Путь для сохранения Parquet в S3 (например, s3a://my-bucket/clean/fraud_data)")
    parser.add_argument("--log-level", default="WARN", choices=["ERROR", "WARN", "INFO", "DEBUG"])

    args = parser.parse_args()

    # Инициализация Spark
    spark = SparkSession.builder \
        .appName("FraudDataCleaning-S3") \
        .getOrCreate()

    spark.sparkContext.setLogLevel(args.log_level)

    # Схема (с сохранением опечатки tranaction_id)
    schema = StructType([
        StructField("tranaction_id", StringType(), True),
        StructField("tx_datetime", StringType(), True),
        StructField("customer_id", StringType(), True),
        StructField("terminal_id", StringType(), True),
        StructField("tx_amount", StringType(), True),
        StructField("tx_time_seconds", StringType(), True),
        StructField("tx_time_days", StringType(), True),
        StructField("tx_fraud", StringType(), True),
        StructField("tx_fraud_scenario", StringType(), True)
    ])

    # === 1. Загрузка всех подходящих файлов из S3 ===
    print(f"📥 Чтение данных из: {args.input}")
    df_raw = spark.read \
        .option("sep", ",") \
        .option("header", "false") \
        .option("mode", "PERMISSIVE") \
        .option("columnNameOfCorruptRecord", "_corrupt_record") \
        .schema(schema) \
        .csv(args.input)

    # Пропуск строк с комментариями (начинаются с '#')
    if "_corrupt_record" in df_raw.columns:
        # Spark помещает строки, не соответствующие схеме, в _corrupt_record
        # В нашем случае — строки вида "# ...", которые не парсятся как 9 полей
        corrupt_count = df_raw.filter(col("_corrupt_record").isNotNull()).count()
        if corrupt_count > 0:
            print(f"🗑️ Пропущено {corrupt_count} строк с комментариями или ошибками формата.")
        df_raw = df_raw.filter(col("_corrupt_record").isNull()).drop("_corrupt_record")

    # === 2. Преобразование типов ===
    df = df_raw \
        .withColumn("tranaction_id", col("tranaction_id").cast(LongType())) \
        .withColumn("customer_id", col("customer_id").cast(LongType())) \
        .withColumn("terminal_id", col("terminal_id").cast(LongType())) \
        .withColumn("tx_amount", col("tx_amount").cast(DoubleType())) \
        .withColumn("tx_time_seconds", col("tx_time_seconds").cast(LongType())) \
        .withColumn("tx_time_days", col("tx_time_days").cast(LongType())) \
        .withColumn("tx_fraud", col("tx_fraud").cast(IntegerType())) \
        .withColumn("tx_fraud_scenario", col("tx_fraud_scenario").cast(IntegerType())) \
        .withColumn("tx_datetime", to_timestamp(col("tx_datetime"), "yyyy-MM-dd HH:mm:ss"))

    # === 3. Фильтрация: удаление строк с NULL после приведения типов ===
    initial_count = df.count()
    df = df.filter(
        col("tranaction_id").isNotNull() &
        col("customer_id").isNotNull() &
        col("terminal_id").isNotNull() &
        col("tx_amount").isNotNull() &
        col("tx_time_seconds").isNotNull() &
        col("tx_time_days").isNotNull() &
        col("tx_fraud").isNotNull() &
        col("tx_fraud_scenario").isNotNull() &
        col("tx_datetime").isNotNull()
    )
    after_type_filter = df.count()
    print(f"🧹 Удалено {initial_count - after_type_filter} строк с некорректными типами.")

    # === 4. Бизнес-правила ===
    df = df.filter(
        (col("tx_amount") > 0) &
        (col("tx_time_seconds") >= 0) &
        (col("tx_time_days") >= 0) &
        (col("tranaction_id") >= 0) &
        (col("customer_id") >= 0) &
        (col("terminal_id") >= 0) &
        (col("tx_fraud").isin([0, 1])) &
        (col("tx_fraud_scenario") >= 0)
    )

    # === 5. Логика: мошенничество → сценарий > 0 ===
    df = df.filter(
        (col("tx_fraud") == 0) | (col("tx_fraud_scenario") >= 1)
    )

    # === 6. Удаление дубликатов по tranaction_id ===
    df_clean = df.dropDuplicates(["tranaction_id"])
    final_count = df_clean.count()
    print(f"✅ Итоговое число строк: {final_count}")

    # === 7. Сохранение в Parquet в S3 ===
    print(f"📤 Сохранение очищенных данных в: {args.output}")
    df_clean.write \
        .mode("overwrite") \
        .option("compression", "snappy") \
        .parquet(args.output)

    print("✨ Очистка завершена успешно.")
    spark.stop()


if __name__ == "__main__":
    main()