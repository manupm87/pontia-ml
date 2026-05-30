# Optimización de hiperparámetros

Métrica optimizada: **roc_auc** · CV de **3** particiones.

| Modelo | busqueda | combinaciones | cv_roc_auc | segundos |
|---|---|---|---|---|
| XGBoost | Randomized | 12 | 0.9501 | 53.8 |
| Random Forest | Randomized | 12 | 0.9345 | 52.7 |
| Decision Tree | Grid | 24 | 0.9187 | 8.4 |
| Logistic Regression | Grid | 8 | 0.8927 | 11.3 |

## Mejores hiperparámetros por modelo

- **Logistic Regression**: `{'C': 10.0, 'class_weight': 'balanced'}`
- **Decision Tree**: `{'criterion': 'entropy', 'max_depth': None, 'min_samples_leaf': 50}`
- **Random Forest**: `{'n_estimators': 300, 'min_samples_leaf': 5, 'max_features': 'sqrt', 'max_depth': 18}`
- **XGBoost**: `{'subsample': 0.9, 'n_estimators': 600, 'max_depth': 16, 'learning_rate': 0.03, 'colsample_bytree': 1.0}`
