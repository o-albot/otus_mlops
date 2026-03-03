#!/usr/bin/env python3
"""
Потоковая инференс-обработка: Kafka → MLflow модель → Kafka
Читает из fraud-transactions-input, применяет модель, пишет в fraud-transactions-predictions
"""
import sys
import os
import json
import time
import argparse
import traceback
import urllib.request
from datetime import datetime
from typing import Dict, Any, Optional

# === Early logging ===
def early_log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

early_log(">>> STREAMING INFERENCE STARTED")

# === Парсинг аргументов ===
try:
    parser = argparse.ArgumentParser()
    # Kafka
    parser.add_argument("--kafka-bootstrap-servers", required=True)
    parser.add_argument("--input-topic", required=True)
    parser.add_argument("--output-topic", required=True)
    parser.add_argument("--security-protocol", default="SASL_SSL")
    parser.add_argument("--sasl-mechanism", default="SCRAM-SHA-512")
    parser.add_argument("--sasl-username", required=True)
    parser.add_argument("--sasl-password", required=True)
    
    # MLflow
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--model-alias", default="champion")
    
    # S3
    parser.add_argument("--s3-bucket-name", required=True)
    parser.add_argument("--s3-endpoint-url", required=True)
    parser.add_argument("--s3-access-key", required=True)
    parser.add_argument("--s3-secret-key", required=True)
    
    # Прочее
    parser.add_argument("--checkpoint-location", default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--read-mode", choices=['latest', 'earliest'], default='latest',
                        help="latest=только новые, earliest=все сообщения в топике")
    
    args = parser.parse_args()
    early_log(f"✓ Аргументы распарсены (read_mode={args.read_mode})")
except Exception as e:
    early_log(f"❌ ARGPARSE ERROR: {e}")
    sys.exit(1)

# === SparkSession с S3-конфигами ===
try:
    from pyspark.sql import SparkSession, Row
    spark = SparkSession.builder \
        .appName("KafkaStreamingInference") \
        .config("spark.hadoop.fs.s3a.endpoint", args.s3_endpoint_url) \
        .config("spark.hadoop.fs.s3a.access.key", args.s3_access_key) \
        .config("spark.hadoop.fs.s3a.secret.key", args.s3_secret_key) \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "true") \
        .getOrCreate()
    early_log("✓ SparkSession создан")
except Exception as e:
    early_log(f"❌ SPARK ERROR: {e}")
    sys.exit(1)

logs = [f"START: {datetime.now().isoformat()}"]
def log(msg: str):
    entry = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    logs.append(entry)
    print(entry, flush=True)

# === Глобальные переменные для graceful shutdown ===
consumer = None
producer = None
start_time = None

try:
    # === Скачиваем сертификат YC ===
    cert_path = "/tmp/CA.pem"
    try:
        urllib.request.urlretrieve("https://storage.yandexcloud.net/cloud-certs/CA.pem", cert_path)
        log(f"✓ Сертификат: {cert_path}")
    except:
        cert_path = "/etc/ssl/certs/ca-certificates.crt"
        log(f"⚠ Используем системный сертификат")
    
    # === S3 переменные ДО импорта mlflow ===
    log("→ Настройка переменных окружения для S3...")
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = args.s3_endpoint_url
    os.environ['AWS_ACCESS_KEY_ID'] = args.s3_access_key
    os.environ['AWS_SECRET_ACCESS_KEY'] = args.s3_secret_key
    log("✓ S3 переменные установлены")
    
    # === Импорт mlflow ===
    log("→ Импорт MLflow...")
    import mlflow
    import mlflow.spark
    log("✓ MLflow импортирован")
    
    # === Настройка MLflow ===
    log("→ Настройка MLflow tracking URI...")
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    log(f"✓ MLflow: {args.tracking_uri}, эксперимент: {args.experiment_name}")
    
    # === Загрузка модели ===
    model_name = f"{args.experiment_name}_model"
    model_uri = f"models:/{model_name}@{args.model_alias}"
    log(f"→ Загрузка модели: {model_uri}")
    
    model = mlflow.spark.load_model(model_uri)
    log(f"✓ Модель загружена: {type(model).__name__}")
    
    # === Kafka Consumer ===
    log("→ Создание KafkaConsumer...")
    from kafka import KafkaConsumer, KafkaProducer
    
    consumer = KafkaConsumer(
        args.input_topic,
        bootstrap_servers=args.kafka_bootstrap_servers,
        security_protocol=args.security_protocol,
        sasl_mechanism=args.sasl_mechanism,
        sasl_plain_username=args.sasl_username,
        sasl_plain_password=args.sasl_password,
        ssl_cafile=cert_path,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        group_id=f"inference-{args.model_alias}-{int(time.time())}",
        auto_offset_reset=args.read_mode,  # ← Теперь управляется аргументом
        enable_auto_commit=False,
        consumer_timeout_ms=30000,  # Увеличили до 30 сек
        max_poll_records=50,  # Батчинг для производительности
    )
    
    # Получаем информацию о топике для отладки
    partitions = consumer.partitions_for_topic(args.input_topic)
    log(f"✓ Consumer: топик '{args.input_topic}', партиций: {len(partitions) if partitions else 'N/A'}")
    log(f"  Режим чтения: {args.read_mode} (earliest=все сообщения, latest=только новые)")
    
    # === Kafka Producer ===
    log("→ Создание KafkaProducer...")
    producer = KafkaProducer(
        bootstrap_servers=args.kafka_bootstrap_servers,
        security_protocol=args.security_protocol,
        sasl_mechanism=args.sasl_mechanism,
        sasl_plain_username=args.sasl_username,
        sasl_plain_password=args.sasl_password,
        ssl_cafile=cert_path,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode('utf-8'),
        acks='all',
        request_timeout_ms=30000,
        retries=3,
    )
    log(f"✓ Producer: топик '{args.output_topic}'")
    
    # === Основной цикл ===
    log("→ Запуск цикла инференса...")
    start_time = time.time()
    
    processed = 0
    errors = 0
    batch_count = 0
    
    for message in consumer:
        # Таймаут
        if args.timeout_seconds > 0 and (time.time() - start_time) > args.timeout_seconds:
            log(f"⏱ Таймаут ({args.timeout_seconds} сек), завершаем")
            break
        
        try:
            input_data = message.value
            
            # transaction_id
            tx_id = input_data.get('transaction_id', f"unknown_{message.offset}")
            
            # Инференс: создаём DataFrame из одной строки
            input_df = spark.createDataFrame([Row(**input_data)])
            predictions = model.transform(input_df)
            result = predictions.select("prediction", "probability").first()
            
            # Извлекаем результат
            prediction = int(result.prediction) if result.prediction is not None else -1
            probability = None
            if result.probability is not None:
                prob_array = result.probability.toArray()
                probability = float(prob_array[1]) if len(prob_array) > 1 else float(prob_array[0])
            
            # Формируем выходное сообщение
            output_message = {
                "transaction_id": tx_id,
                "original_timestamp": input_data.get("timestamp"),
                "processed_at": datetime.now().isoformat(),
                "prediction": prediction,
                "probability": probability,
                "model_alias": args.model_alias,
                "input_features": {k: v for k, v in input_data.items() if k not in ['fraud', 'label']}
            }
            
            # Отправляем в output topic
            future = producer.send(args.output_topic, value=output_message)
            # Ждём подтверждения для надёжности (можно убрать для производительности)
            future.get(timeout=10)
            
            processed += 1
            batch_count += 1
            
            # Коммит и прогресс
            if batch_count >= args.batch_size:
                consumer.commit()
                log(f"✓ Обработано {processed} сообщений, коммит оффсетов")
                batch_count = 0
                
            if processed % 100 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                log(f"  Прогресс: {processed} сообщений, {rate:.1f} msg/sec")
                
        except Exception as e:
            errors += 1
            log(f"⚠ Ошибка offset={message.offset}: {type(e).__name__}: {str(e)[:100]}")
            # Продолжаем обработку
    
    # === Финализация ===
    log("→ Завершение: коммит оффсетов и flush producer...")
    if consumer:
        consumer.commit()
    if producer:
        producer.flush(timeout=30)
    
    elapsed = time.time() - start_time
    log(f"✓✓✓ INFERENCE COMPLETED ✓✓✓")
    log(f"  • Обработано сообщений: {processed}")
    log(f"  • Ошибок: {errors}")
    log(f"  • Время работы: {elapsed:.1f} сек")
    if elapsed > 0:
        log(f"  • Скорость: {processed/elapsed:.1f} msg/sec")
    log(f"  • Результат записан в топик: {args.output_topic}")
    
    # Если сообщений 0 — добавляем подсказку в лог
    if processed == 0:
        log("⚠ Подсказка: если сообщений 0, проверьте:")
        log("  • Параметр --read-mode: 'latest' читает только новые сообщения")
        log("  • Параметр --read-mode: 'earliest' читает все сообщения в топике")
        log("  • Есть ли данные в топике fraud-transactions-input?")
    
except Exception as e:
    log(f"❌ FATAL ERROR: {type(e).__name__}: {str(e)[:200]}")
    for line in traceback.format_exc().split('\n')[-8:]:
        if line.strip():
            log(f"  {line}")

finally:
    # === Логи в S3 ===
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_df = spark.createDataFrame([Row(message=l) for l in logs])
        log_path = f"s3a://{args.s3_bucket_name}/debug_logs/kafka_inference_{ts}"
        log_df.write.mode("overwrite").text(log_path)
        log(f"✓ Логи сохранены в S3: {log_path}")
    except Exception as e:
        print(f"⚠ Не удалось записать логи в S3: {e}", flush=True)
    
    # === Очистка ===
    try:
        if producer:
            producer.close(timeout=10)
            log("✓ Producer закрыт")
    except:
        pass
    try:
        if consumer:
            consumer.close()
            log("✓ Consumer закрыт")
    except:
        pass
    try:
        spark.stop()
        log("✓ SparkSession остановлен")
    except:
        pass
    
    print(">>> STREAMING INFERENCE FINISHED <<<", flush=True)
    sys.exit(0)