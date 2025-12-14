import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import streamlit as st
import joblib
from datetime import datetime

def prepare_data(data):
    try:
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]), numeric_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
                ]), categorical_features)
            ] if len(categorical_features) > 0 else [
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]), numeric_features)
            ]
        )
        
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)
            
        return preprocessor, X, y
        
    except Exception as e:
        st.error(f"Veri hazırlama hatası: {str(e)}")
        return None, None, None

def process_tabular_data(data, task_type):
    try:
        preprocessor, X, y = prepare_data(data)
        if X is None:
            return None, 0, None, None

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        models = {
            'regression': {
                'RandomForest': (RandomForestRegressor(), {
                    'n_estimators': [50, 100],
                    'max_depth': [None, 10]
                }),
                'DecisionTree': (DecisionTreeRegressor(), {
                    'max_depth': [None, 10],
                    'min_samples_split': [2, 5]
                }),
                'SVM': (SVR(), {
                    'kernel': ['rbf', 'linear'],
                    'C': [0.1, 1, 10]
                })
            },
            'classification': {
                'RandomForest': (RandomForestClassifier(), {
                    'n_estimators': [50, 100],
                    'max_depth': [None, 10]
                }),
                'DecisionTree': (DecisionTreeClassifier(), {
                    'max_depth': [None, 10],
                    'min_samples_split': [2, 5]
                }),
                'SVM': (SVC(probability=True), {
                    'kernel': ['rbf', 'linear'],
                    'C': [0.1, 1, 10]
                })
            }
        }

        best_score = float('-inf')
        best_model = None
        best_name = None

        for name, (model, params) in models[task_type].items():
            st.write(f"\n{name} eğitiliyor...")

            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            pipeline.fit(X_train, y_train)
            score = pipeline.score(X_test, y_test)

            st.write(f"Test skoru: {score:.4f}")

            if score > best_score:
                best_score = score
                best_model = pipeline
                best_name = name

        if best_model:
            model_path = f"models/{best_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            joblib.dump(best_model, model_path)
            st.success(f"\nEn iyi model: {best_name}")

        return best_name, best_score, model_path, best_name

    except Exception as e:
        st.error(f"Hata: {str(e)}")
        return None, 0, None, None

        
    except Exception as e:
        st.error(f"İşlem hatası: {str(e)}")
        return None, 0, None, None