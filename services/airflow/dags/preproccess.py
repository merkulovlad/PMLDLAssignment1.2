from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

RAW_DATA_PATH = Path(__file__).resolve().parents[3] / 'data/raw/titanic.csv'
PROCESSED_DIR = Path(__file__).resolve().parents[3] / 'data/processed'

def process_data(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    cat_cols=('sex', 'embarked'),
    num_impute_strategy='median',
    cat_impute_strategy='most_frequent',
):
    x_train = x_train.copy()
    x_test  = x_test.copy()
    cat_cols = [c for c in cat_cols if c in x_train.columns]  # keep only existing
    num_cols = [c for c in x_train.columns if c not in cat_cols]

    # 0) Coerce numeric columns to numbers (others left as strings)
    x_train[num_cols] = x_train[num_cols].apply(pd.to_numeric, errors='coerce')
    x_test[num_cols]  = x_test[num_cols].apply(pd.to_numeric, errors='coerce')

    # 1) Impute numerics
    num_imputer = SimpleImputer(strategy=num_impute_strategy)
    x_train_num = pd.DataFrame(
        num_imputer.fit_transform(x_train[num_cols]),
        columns=num_cols,
        index=x_train.index,
    )
    x_test_num = pd.DataFrame(
        num_imputer.transform(x_test[num_cols]),
        columns=num_cols,
        index=x_test.index,
    )

    # 2) Impute categoricals
    cat_imputer = SimpleImputer(strategy=cat_impute_strategy)
    x_train_cat = pd.DataFrame(
        cat_imputer.fit_transform(x_train[cat_cols]),
        columns=cat_cols,
        index=x_train.index,
    ) if cat_cols else pd.DataFrame(index=x_train.index)

    x_test_cat = pd.DataFrame(
        cat_imputer.transform(x_test[cat_cols]),
        columns=cat_cols,
        index=x_test.index,
    ) if cat_cols else pd.DataFrame(index=x_test.index)

    # 3) One-Hot Encode cats (handle sklearn version differences)
    if cat_cols:
        try:
            encoder = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)
        except TypeError:
            # older sklearn
            encoder = OneHotEncoder(handle_unknown='ignore', drop='first', sparse=False)

        ohe_train = encoder.fit_transform(x_train_cat)
        ohe_test  = encoder.transform(x_test_cat)
        ohe_cols  = encoder.get_feature_names_out(cat_cols)

        x_train_ohe = pd.DataFrame(ohe_train, columns=ohe_cols, index=x_train.index)
        x_test_ohe  = pd.DataFrame(ohe_test,  columns=ohe_cols,  index=x_test.index)
    else:
        encoder = None
        x_train_ohe = pd.DataFrame(index=x_train.index)
        x_test_ohe  = pd.DataFrame(index=x_test.index)

    # 4) Concatenate numerics + OHE (now everything is numeric)
    x_train_final = pd.concat([x_train_num, x_train_ohe], axis=1)
    x_test_final  = pd.concat([x_test_num,  x_test_ohe],  axis=1)

    # 5) Scale
    scaler = MinMaxScaler()
    x_train_scaled = pd.DataFrame(
        scaler.fit_transform(x_train_final),
        columns=x_train_final.columns,
        index=x_train_final.index,
    )
    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test_final),
        columns=x_test_final.columns,
        index=x_test_final.index,
    )

    return x_train_scaled, x_test_scaled, num_imputer, cat_imputer, encoder, scaler


def run_preprocessing(**context):
    raw_df = pd.read_csv(RAW_DATA_PATH)

    feature_cols = [c for c in raw_df.columns if c not in {'survived', 'name'}]
    x = raw_df[feature_cols]
    y = raw_df['survived']

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    x_train_processed, x_test_processed, *artifacts = process_data(x_train, x_test)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_out = x_train_processed.copy()
    train_out['survived'] = y_train.values
    train_out.to_csv(PROCESSED_DIR / 'train.csv', index=False)

    test_out = x_test_processed.copy()
    test_out['survived'] = y_test.values
    test_out.to_csv(PROCESSED_DIR / 'test.csv', index=False)

    artifact_path = PROCESSED_DIR / 'preprocess_pipeline.joblib'
    joblib.dump(
        {
            'num_imputer': artifacts[0],
            'cat_imputer': artifacts[1],
            'encoder': artifacts[2],
            'scaler': artifacts[3],
        },
        artifact_path,
    )

    return {
        'train_path': str(PROCESSED_DIR / 'train.csv'),
        'test_path': str(PROCESSED_DIR / 'test.csv'),
        'pipeline_path': str(artifact_path),
    }

with DAG('process_data', start_date=datetime(2025, 1, 1), schedule_interval='@daily') as dag:
    preprocess_data_task = PythonOperator(
        task_id='process_data',
        python_callable=run_preprocessing,
    )
