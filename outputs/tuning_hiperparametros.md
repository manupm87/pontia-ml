# Optimización de hiperparámetros

Métrica optimizada: **roc_auc** · CV de **3** particiones.

| Modelo | busqueda | combinaciones | cv_roc_auc_default | cv_roc_auc_tuned | mejora | segundos |
|---|---|---|---|---|---|---|
| XGBoost | RandomizedSearchCV | 12 | 0.9573 | 0.9586 | 0.0013 | 53.8 |
| Random Forest | RandomizedSearchCV | 12 | 0.9372 | 0.9439 | 0.0066 | 53.0 |
| Decision Tree | GridSearchCV | 24 | 0.9259 | 0.9296 | 0.0036 | 10.1 |
| Logistic Regression | GridSearchCV | 8 | 0.9077 | 0.9082 | 0.0005 | 16.7 |

## Mejores hiperparámetros por modelo

- **Logistic Regression**: `{'C': 10.0, 'class_weight': 'balanced'}`
- **Decision Tree**: `{'criterion': 'gini', 'max_depth': None, 'min_samples_leaf': 50}`
- **Random Forest**: `{'n_estimators': 300, 'min_samples_leaf': 5, 'max_features': 'sqrt', 'max_depth': 18}`
- **XGBoost**: `{'subsample': 0.9, 'n_estimators': 600, 'max_depth': 16, 'learning_rate': 0.03, 'colsample_bytree': 1.0}`
