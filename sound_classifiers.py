# https://airflow.apache.org/docs/apache-airflow/stable/tutorial/fundamentals.html
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

RUN_DIR = os.environ['SC_RUN_DIR']

with DAG(
    'sound_classifiers',
    default_args={
        'depends_on_past': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
    description='Train sound classifiers',
    schedule=timedelta(days=1),
    start_date=datetime(2021, 1, 1),
    catchup=True,
) as dag:
    t1 = BashOperator(
        task_id='preprocess',
        bash_command=f'cd {RUN_DIR} && python3 main.py preprocess_data',
    )

    t2 = BashOperator(
        task_id='train_rbf_svm',
        depends_on_past=True,
        bash_command=f'cd {RUN_DIR} && python3 main.py train_rbf_svm',
    )

    t3 = BashOperator(
        task_id='train_linear_svm',
        depends_on_past=True,
        bash_command=f'cd {RUN_DIR} && python3 main.py train_linear_svm',
    )

    '''
    t4 = BashOperator(
        task_id='train_cnn',
        depends_on_past=True,
        bash_command=f'cd {RUN_DIR} && python3 main.py train_cnn',
    )
    '''

    t1 >> [t2, t3]
