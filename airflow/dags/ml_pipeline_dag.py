from airflow import DAG
from airflow.operators.python import PythonOperator

import os
import pandas as pd
import logging
import pickle
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn


parent_directory = os.path.dirname(os.path.abspath(__file__))


def load_data():
    '''
    loads input data for model
    '''
    df = pd.read_csv(os.path.join(parent_directory, 'data/processed/', 'final_data.csv'))

    # check for null values
    null_percentages = df.isnull().mean() * 100
    for feature, percentage in null_percentages.items():
        if percentage > 5:
            logging.warning(f"Feature '{feature}' has {percentage:.2f}% missing values.")

    # check for duplicates
    num_dupes = df.duplicated().sum()
    if num_dupes > 0:
        logging.warning(f"Dataset contains {num_dupes} duplicate records.")

    # check for imbalanced classes
    churn_dist = df['churn'].value_counts(normalize=True)
    if churn_dist.min() < 0.05 or churn_dist.max() > 0.95:
        logging.warning("Imbalanced classes detected in target variable 'churn'.")

    return df


def train_model(**context):
    '''
    trains model and saves as pickle
    '''
    df = context['task_instance'].xcom_pull(task_ids='load_data')

    X_train, X_test, y_train, y_test = train_test_split(df.drop('churn', axis=1),
                                                        df['churn'],
                                                        test_size=0.2,
                                                        random_state=42)

    context['task_instance'].xcom_push(key='X_test', value=X_test.values.tolist())
    context['task_instance'].xcom_push(key='y_test', value=y_test.values.tolist())

    rf = RandomForestClassifier(max_depth=None,
                                min_samples_leaf=4,
                                min_samples_split=5,
                                n_estimators=300,
                                random_state=42
                                )
    xgb = XGBClassifier(learning_rate=0.1,
                        max_depth=5,
                        n_estimators=200,
                        random_state=42,
                        )

    model = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb)], voting='soft')

    model.fit(X_train, y_train)

    with open(os.path.join(parent_directory, 'models', 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)

    mlflow.set_tracking_uri('http://mlflow:5000')
    mlflow.set_experiment('churn_predictor')

    with mlflow.start_run():
        mlflow.sklearn.log_model(model, 'churn_classification_model')
        for model_name, model_type in model.named_estimators_.items():
            for param_name, param_value in model_type.get_params().items():
                mlflow.log_param(f'{model_name}_{param_name}', param_value)
        mlflow.log_param('voting', model.voting)


def evaluate_model(**context):
    '''
    loads trained model, evaluates on test set, logs metrics
    '''
    with open(os.path.join(parent_directory, 'models', 'model.pkl'), 'rb') as f:
        model = pickle.load(f)

    X_test = context['task_instance'].xcom_pull(task_ids='train_model', key='X_test')
    y_test = context['task_instance'].xcom_pull(task_ids='train_model', key='y_test')

    y_pred = model.predict(X_test)

    context['task_instance'].xcom_push(key='y_pred', value=y_pred.tolist())

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logging.info(f'Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}')

    mlflow.set_tracking_uri('http://mlflow:5000')
    mlflow.set_experiment('churn_predictor')

    with mlflow.start_run():
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('f1_score', f1)


def generate_report(**context):
    '''
    output classification report and confusion matrix to reporting folder
    '''
    reporting_dir = os.path.join(parent_directory, 'reporting')

    y_test = context['task_instance'].xcom_pull(task_ids='train_model', key='y_test')
    y_pred = context['task_instance'].xcom_pull(task_ids='evaluate_model', key='y_pred')

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # write classification report to txt file
    report_path = os.path.join(reporting_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(str(report))

    # write confusion matrix to png
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=True)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(os.path.join(reporting_dir, 'confusion_matrix.png'))

    mlflow.set_tracking_uri('http://mlflow:5000')
    mlflow.set_experiment('churn_predictor')

    with mlflow.start_run():
        mlflow.log_artifact(os.path.join(reporting_dir, 'confusion_matrix.png'))


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1
}


dag = DAG(
    'model_pipeline',
    default_args=default_args,
    description='A DAG for model training and evaluation',
    schedule_interval=None,
    start_date=datetime.now(),
    tags=['model-training']
)


load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    provide_context=True,
    dag=dag
)


train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    provide_context=True,
    dag=dag
)

evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    provide_context=True,
    dag=dag
)

generate_report_task = PythonOperator(
    task_id='generate_report',
    python_callable=generate_report,
    provide_context=True,
    dag=dag
)


load_data_task >> train_model_task >> evaluate_model_task >> generate_report_task
