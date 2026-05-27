# Optimización de hiperparámetros

Métrica optimizada: **roc_auc** · CV de **3** particiones.

| Modelo | busqueda | combinaciones | cv_roc_auc_default | cv_roc_auc_tuned | mejora | segundos |
|---|---|---|---|---|---|---|
| XGBoost | RandomizedSearchCV | 12 | 0.9542 | 0.958 | 0.0038 | 18.9 |
| Random Forest | RandomizedSearchCV | 12 | 0.9408 | 0.9469 | 0.0061 | 56.6 |
| Decision Tree | GridSearchCV | 24 | 0.9311 | 0.9341 | 0.003 | 8.6 |
| Logistic Regression | GridSearchCV | 8 | 0.9093 | 0.9097 | 0.0004 | 12.1 |

## Mejores hiperparámetros por modelo

- **Logistic Regression**: `{'C': 10.0, 'class_weight': 'balanced'}`
- **Decision Tree**: `{'criterion': 'entropy', 'max_depth': None, 'min_samples_leaf': 50}`
- **Random Forest**: `{'n_estimators': 300, 'min_samples_leaf': 5, 'max_features': 'sqrt', 'max_depth': 18}`
- **XGBoost**: `{'subsample': 0.9, 'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.2, 'colsample_bytree': 1.0}`
