#!/usr/bin/env python3
"""
Генератор данных в Kafka: отправка всех записей из датасета
"""
from pyspark.sql import SparkSession
from datetime import datetime
import sys
import urllib.request
import json
import argparse

# Парсинг аргументов
parser = argparse.ArgumentParser()
parser.add_argument("--input-file", required=True, help="Путь к данным в S3")
parser.add_argument("--bootstrap-servers", required=True, help="Адрес кластера Kafka")
parser.add_argument("--topic", required=True, help="Топик Kafka для отправки")
parser.add_argument("--security-protocol", default="SASL_SSL", help="Протокол безопасности")
parser.add_argument("--sasl-mechanism", default="SCRAM-SHA-512", help="Механизм SASL")
parser.add_argument("--sasl-username", required=True, help="Имя пользователя SASL")
parser.add_argument("--sasl-password", required=True, help="Пароль SASL")
parser.add_argument("--s3-endpoint-url", required=True, help="S3 endpoint URL")
parser.add_argument("--s3-access-key", required=True, help="S3 access key")
parser.add_argument("--s3-secret-key", required=True, help="S3 secret key")
parser.add_argument("--s3-bucket-name", required=True, help="Имя бакета для логов")
args = parser.parse_args()

# Создание SparkSession
spark = SparkSession.builder.appName("KafkaDataProducer").getOrCreate()

logs = [
    f"START: {datetime.now().isoformat()}",
    f"Spark version: {spark.version}",
    f"Input file: {args.input_file}",
    f"Kafka servers: {args.bootstrap_servers}",
    f"Topic: {args.topic}",
]

try:
    # Шаг 1: Скачиваем сертификат Yandex Cloud
    logs.append("→ Скачиваем сертификат Yandex Cloud...")
    cert_url = "https://storage.yandexcloud.net/cloud-certs/CA.pem"
    cert_path = "/tmp/CA.pem"
    
    try:
        urllib.request.urlretrieve(cert_url, cert_path)
        logs.append(f"✓ Сертификат скачан: {cert_path}")
    except Exception as e:
        logs.append(f"⚠ Не удалось скачать сертификат: {type(e).__name__}")
        logs.append(f"  Используем системный сертификат")
        cert_path = "/etc/ssl/certs/ca-certificates.crt"
    
    # Шаг 2: Импорт после создания сессии
    from kafka import KafkaProducer
    logs.append("✓ Kafka imported successfully")
    
    # Шаг 3: Конфигурация продюсера
    config = {
        'bootstrap_servers': args.bootstrap_servers,
        'security_protocol': args.security_protocol,
        'sasl_mechanism': args.sasl_mechanism,
        'sasl_plain_username': args.sasl_username,
        'sasl_plain_password': args.sasl_password,
        'ssl_cafile': cert_path,
        'value_serializer': lambda v: json.dumps(v, ensure_ascii=False).encode('utf-8'),
        'request_timeout_ms': 30000,
        'retry_backoff_ms': 1000,
        'max_block_ms': 60000,
        'acks': 'all',
    }
    
    logs.append("→ Создание KafkaProducer...")
    producer = KafkaProducer(**config)
    logs.append("✓ KafkaProducer created successfully")
    
    # Шаг 4: Загрузка данных из S3
    logs.append(f"→ Загрузка данных из: {args.input_file}")
    df = spark.read.csv(args.input_file, header=True, inferSchema=True)
    
    # Убираем целевую переменную 'fraud' для потоковых данных
    if 'fraud' in df.columns:
        df = df.drop('fraud')
        logs.append("✓ Целевая переменная 'fraud' удалена из данных")
    
    # Получаем общее количество записей
    total_records = df.count()
    logs.append(f"✓ Загружено {total_records} записей для отправки в Kafka")
    
    # Шаг 5: Отправка всех записей в Kafka
    logs.append("→ Начинаем отправку данных в Kafka...")
    records = [row.asDict() for row in df.collect()]
    
    sent = 0
    batch_size = 100
    for i, record in enumerate(records):
        producer.send(args.topic, value=record)
        sent += 1
        
        # Прогресс каждые 100 записей или на последней записи
        if (i + 1) % batch_size == 0 or i == len(records) - 1:
            logs.append(f"  Отправлено {i + 1} из {total_records} записей ({(i + 1) / total_records * 100:.1f}%)")
    
    # Ждём подтверждения доставки всех сообщений
    logs.append("→ Ожидание подтверждения доставки всех сообщений...")
    producer.flush(timeout=120)
    
    logs.append(f"✓ SUCCESS: все {sent} сообщений успешно отправлены в топик '{args.topic}'")
    logs.append("✓✓✓ KAFKA DATA PRODUCTION COMPLETED ✓✓✓")
    
except Exception as e:
    logs.append(f"❌ CRITICAL_ERROR: {type(e).__name__}: {str(e)[:250]}")
    import traceback
    for line in traceback.format_exc().split('\n')[-10:]:
        if line.strip():
            logs.append(f"  {line}")
    sys.exit(1)
    
finally:
    # Сохранение логов в S3
    from pyspark.sql import Row
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_df = spark.createDataFrame([Row(message=log) for log in logs])
    log_path = f"s3a://{args.s3_bucket_name}/debug_logs/kafka_producer_{timestamp}"
    log_df.write.mode("overwrite").text(log_path)
    logs.append(f"✓ Логи сохранены в S3: {log_path}")
    
    # Вывод в консоль для отладки
    for log in logs[-15:]:
        print(log, flush=True)
    
    # Закрытие ресурсов
    try:
        producer.close(timeout=10)
        logs.append("✓ Kafka producer закрыт")
    except:
        pass
    
    spark.stop()
    logs.append("✓ SparkSession остановлен")
    print("✓✓✓ ЗАДАЧА ЗАВЕРШЕНА ✓✓✓", flush=True)