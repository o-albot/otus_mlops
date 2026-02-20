#!/usr/bin/env python3
"""
A/B тестирование моделей: сравнение champion vs challenger из реестра MLflow
"""
import sys
import os
import argparse
from pyspark.sql import SparkSession
from datetime import datetime
import numpy as np
from scipy.stats import ttest_ind

# Парсинг аргументов
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Путь к данным")
parser.add_argument("--s3-endpoint-url", required=True, help="S3 endpoint URL")
parser.add_argument("--s3-access-key", required=True, help="S3 access key")
parser.add_argument("--s3-secret-key", required=True, help="S3 secret key")
parser.add_argument("--tracking-uri", required=True, help="MLflow tracking URI")
parser.add_argument("--s3-bucket-name", required=True, help="S3 bucket name")
parser.add_argument("--experiment-name", default="fraud_detection", help="MLflow experiment name")
args = parser.parse_args()

spark = SparkSession.builder.appName("ABTestMLflowRegistry").getOrCreate()

logs = [
    "=" * 60,
    "A/B ТЕСТИРОВАНИЕ МОДЕЛЕЙ (MLflow Registry)",
    "=" * 60,
    f"Spark version: {spark.version}",
    f"Input path: {args.input}",
    f"Tracking URI: {args.tracking_uri}",
    f"S3 bucket: {args.s3_bucket_name}",
    ""
]

try:
    # === 1. Загрузка тестовых данных ===
    logs.append("=== ШАГ 1: ЗАГРУЗКА ДАННЫХ ===")
    logs.append(f"Загрузка данных: {args.input}")
    
    df = spark.read.csv(args.input, header=True, inferSchema=True)
    _, test_df = df.randomSplit([0.8, 0.2], seed=42)
    test_size = test_df.count()
    logs.append(f"✓ Тестовая выборка: {test_size} записей")
    
    # === 2. Настройка окружения ===
    logs.append("\n=== ШАГ 2: НАСТРОЙКА ОКРУЖЕНИЯ ===")
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = args.s3_endpoint_url
    os.environ['AWS_ACCESS_KEY_ID'] = args.s3_access_key
    os.environ['AWS_SECRET_ACCESS_KEY'] = args.s3_secret_key
    
    import mlflow
    import mlflow.spark
    from pyspark.ml.evaluation import (
        BinaryClassificationEvaluator,
        MulticlassClassificationEvaluator
    )
    
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    logs.append("✓ MLflow настроен")
    
    # === 3. Загрузка моделей из реестра MLflow ===
    logs.append("\n=== ШАГ 3: ЗАГРУЗКА МОДЕЛЕЙ ИЗ РЕЕСТРА MLFLOW ===")
    
    model_name = f"{args.experiment_name}_model"
    
    # Загрузка champion
    champion_uri = f"models:/{model_name}@champion"
    logs.append(f"Загрузка champion: {champion_uri}")
    champion_model = mlflow.spark.load_model(champion_uri)
    logs.append("✓ Champion модель загружена")
    
    # Загрузка challenger
    challenger_uri = f"models:/{model_name}@challenger"
    logs.append(f"Загрузка challenger: {challenger_uri}")
    challenger_model = mlflow.spark.load_model(challenger_uri)
    logs.append("✓ Challenger модель загружена")
    
    # === 4. Оценка метрик ===
    logs.append("\n=== ШАГ 4: ОЦЕНКА МЕТРИК ===")
    evaluator_auc = BinaryClassificationEvaluator(labelCol="fraud", metricName="areaUnderROC")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="fraud", metricName="f1")
    evaluator_acc = MulticlassClassificationEvaluator(labelCol="fraud", metricName="accuracy")
    
    prod_pred = champion_model.transform(test_df)
    prod_auc = evaluator_auc.evaluate(prod_pred)
    prod_f1 = evaluator_f1.evaluate(prod_pred)
    prod_acc = evaluator_acc.evaluate(prod_pred)
    
    cand_pred = challenger_model.transform(test_df)
    cand_auc = evaluator_auc.evaluate(cand_pred)
    cand_f1 = evaluator_f1.evaluate(cand_pred)
    cand_acc = evaluator_acc.evaluate(cand_pred)
    
    logs.append("Метрики моделей на тестовой выборке:")
    logs.append(f"  Champion - AUC: {prod_auc:.4f} | F1: {prod_f1:.4f} | Acc: {prod_acc:.4f}")
    logs.append(f"  Challenger - AUC: {cand_auc:.4f} | F1: {cand_f1:.4f} | Acc: {cand_acc:.4f}")
    
    # === 5. Статистический анализ ===
    logs.append("\n=== ШАГ 5: СТАТИСТИЧЕСКИЙ АНАЛИЗ ===")
    logs.append("Выполнение bootstrap-анализа (1000 итераций)...")
    
    def bootstrap_f1(preds_champion, preds_candidate, n_iter=1000, random_state=42):
        prod_pd = preds_champion.select("fraud", "prediction").toPandas()
        cand_pd = preds_candidate.select("fraud", "prediction").toPandas()
        prod_f1_scores, cand_f1_scores = [], []
        for i in range(n_iter):
            prod_sample = prod_pd.sample(n=len(prod_pd), replace=True, random_state=random_state + i)
            cand_sample = cand_pd.sample(n=len(cand_pd), replace=True, random_state=random_state + i)
            from sklearn.metrics import f1_score
            prod_f1 = f1_score(prod_sample["fraud"], prod_sample["prediction"], zero_division=0)
            cand_f1 = f1_score(cand_sample["fraud"], cand_sample["prediction"], zero_division=0)
            prod_f1_scores.append(prod_f1)
            cand_f1_scores.append(cand_f1)
        t_stat, p_value = ttest_ind(cand_f1_scores, prod_f1_scores, equal_var=False)
        pooled_std = np.sqrt((np.std(cand_f1_scores)**2 + np.std(prod_f1_scores)**2) / 2)
        effect_size = (np.mean(cand_f1_scores) - np.mean(prod_f1_scores)) / pooled_std
        return {
            "prod_f1_mean": float(np.mean(prod_f1_scores)),
            "prod_f1_std": float(np.std(prod_f1_scores)),
            "cand_f1_mean": float(np.mean(cand_f1_scores)),
            "cand_f1_std": float(np.std(cand_f1_scores)),
            "f1_improvement": float(np.mean(cand_f1_scores) - np.mean(prod_f1_scores)),
            "p_value": float(p_value),
            "effect_size": float(effect_size),
            "is_significant": bool(p_value < 0.05 and np.mean(cand_f1_scores) > np.mean(prod_f1_scores)),
            "bootstrap_iterations": n_iter
        }
    
    bootstrap_results = bootstrap_f1(prod_pred, cand_pred, n_iter=1000, random_state=42)
    
    logs.append(f"Champion F1 (bootstrap): {bootstrap_results['prod_f1_mean']:.4f} ± {bootstrap_results['prod_f1_std']:.4f}")
    logs.append(f"Challenger F1 (bootstrap): {bootstrap_results['cand_f1_mean']:.4f} ± {bootstrap_results['cand_f1_std']:.4f}")
    logs.append(f"Улучшение F1: {bootstrap_results['f1_improvement']:+.4f}")
    logs.append(f"p-value: {bootstrap_results['p_value']:.6f}")
    logs.append(f"Cohen's d (размер эффекта): {bootstrap_results['effect_size']:.4f}")
    logs.append(f"Статистически значимо (α=0.05): {'ДА' if bootstrap_results['is_significant'] else 'НЕТ'}")
    
    # === 6. Принятие решения ===
    logs.append("\n=== ШАГ 6: ПРИНЯТИЕ РЕШЕНИЯ ===")
    decision = "PROMOTE" if bootstrap_results["is_significant"] else "KEEP_CHAMPION"
    logs.append(f"РЕШЕНИЕ: {decision}")
    logs.append(f"Обоснование: {'Challenger показал статистически значимое улучшение' if decision == 'PROMOTE' else 'Challenger не превосходит champion'}")
    
    # === 7. Логирование в MLflow ===
    logs.append("\n=== ШАГ 7: ЛОГИРОВАНИЕ В MLFLOW ===")
    with mlflow.start_run(run_name=f"ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        mlflow.log_params({
            "test_set_size": test_size,
            "input_path": args.input,
            "champion_model": champion_uri,
            "challenger_model": challenger_uri,
            "bootstrap_iterations": bootstrap_results["bootstrap_iterations"],
            "significance_level": 0.05
        })
        
        mlflow.log_metrics({
            "champion_auc": prod_auc,
            "champion_f1": prod_f1,
            "champion_accuracy": prod_acc,
            "challenger_auc": cand_auc,
            "challenger_f1": cand_f1,
            "challenger_accuracy": cand_acc,
            "f1_improvement": bootstrap_results["f1_improvement"],
            "p_value": bootstrap_results["p_value"],
            "effect_size": bootstrap_results["effect_size"]
        })
        
        mlflow.set_tag("ab_test_decision", decision)
        mlflow.set_tag("statistically_significant", str(bootstrap_results["is_significant"]))
        mlflow.set_tag("model_name", model_name)
        
        logs.append(f"✓ MLflow run ID: {run.info.run_id}")
        
        # Автоматическое продвижение challenger в champion при значимом улучшении
        if decision == "PROMOTE":
            logs.append("\n=== ШАГ 8: ПРОДВИЖЕНИЕ CHALLENGER В CHAMPION ===")
            client = mlflow.tracking.MlflowClient()
            
            # Получаем версию challenger
            challenger_version = None
            versions = client.get_latest_versions(model_name)
            for version in versions:
                if hasattr(version, 'aliases') and "challenger" in version.aliases:
                    challenger_version = version.version
                    break
            
            if challenger_version:
                # Устанавливаем алиас champion для challenger
                if hasattr(client, 'set_registered_model_alias'):
                    client.set_registered_model_alias(model_name, "champion", challenger_version)
                else:
                    client.set_model_version_tag(model_name, challenger_version, "alias", "champion")
                
                # Снимаем алиас challenger со старой версии
                for version in versions:
                    if hasattr(version, 'aliases') and "champion" in version.aliases and version.version != challenger_version:
                        if hasattr(client, 'delete_registered_model_alias'):
                            client.delete_registered_model_alias(model_name, "champion")
                        else:
                            client.delete_model_version_tag(model_name, version.version, "alias")
                
                logs.append(f"✓ Версия {challenger_version} продвинута в 'champion'")
                logs.append(f"✓ Старый 'champion' снят с алиаса")
            else:
                logs.append("⚠ Не удалось определить версию challenger для продвижения")
    
    # === 8. Сохранение отчёта в S3 ===
    logs.append("\n=== ШАГ 9: СОХРАНЕНИЕ ОТЧЁТА В S3 ===")
    from pyspark.sql import Row
    
    report = spark.createDataFrame([Row(
        champion_auc=float(prod_auc),
        champion_f1=float(prod_f1),
        champion_accuracy=float(prod_acc),
        challenger_auc=float(cand_auc),
        challenger_f1=float(cand_f1),
        challenger_accuracy=float(cand_acc),
        f1_improvement=float(bootstrap_results["f1_improvement"]),
        p_value=float(bootstrap_results["p_value"]),
        effect_size=float(bootstrap_results["effect_size"]),
        is_significant=bool(bootstrap_results["is_significant"]),
        decision=decision,
        test_set_size=test_size,
        mlflow_run_id=run.info.run_id,
        champion_model=champion_uri,
        challenger_model=challenger_uri,
        timestamp=datetime.now().isoformat()
    )])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"s3a://{args.s3_bucket_name}/models/ab_reports/report_{timestamp}"
    report.write.mode("overwrite").json(report_path)
    logs.append(f"✓ Отчёт сохранён: {report_path}")
    
    # === 9. Сохранение логов в S3 ===
    logs.append("\n" + "=" * 60)
    logs.append("✓✓✓ A/B ТЕСТИРОВАНИЕ ЗАВЕРШЕНО УСПЕШНО!")
    logs.append("=" * 60)
    
    log_df = spark.createDataFrame([Row(message=log) for log in logs])
    log_df.write.mode("overwrite").text(
        f"s3a://{args.s3_bucket_name}/debug_logs/ab_test_{timestamp}"
    )
    logs.append(f"✓ Детальные логи: debug_logs/ab_test_{timestamp}")
    
except Exception as e:
    logs.append(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {type(e).__name__}")
    logs.append(f"Сообщение: {str(e)}")
    import traceback
    logs.append("\nTraceback:")
    for line in traceback.format_exc().split('\n')[-10:]:
        if line.strip():
            logs.append(line)
    raise
finally:
    for msg in logs[-15:]:
        print(msg, flush=True)
    spark.stop()
    print("SparkSession остановлен", flush=True)