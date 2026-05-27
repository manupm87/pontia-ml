# Balanceo de clases — comparación de estrategias

Desbalance del problema: ~37 % de cancelaciones. Métricas sobre el conjunto de test, con hiperparámetros base (para aislar el efecto del balanceo).

| modelo | estrategia | accuracy | precision | recall | f1 | roc_auc |
|---|---|---|---|---|---|---|
| Logistic Regression | baseline | 0.825 | 0.805 | 0.696 | 0.746 | 0.907 |
| Decision Tree | baseline | 0.854 | 0.826 | 0.768 | 0.796 | 0.934 |
| Random Forest | baseline | 0.861 | 0.887 | 0.717 | 0.793 | 0.943 |
| XGBoost | baseline | 0.883 | 0.857 | 0.819 | 0.838 | 0.955 |
| Logistic Regression | class_weight | 0.822 | 0.735 | 0.813 | 0.772 | 0.908 |
| Decision Tree | class_weight | 0.836 | 0.732 | 0.878 | 0.799 | 0.929 |
| Random Forest | class_weight | 0.868 | 0.813 | 0.837 | 0.825 | 0.944 |
| XGBoost | class_weight | 0.877 | 0.806 | 0.878 | 0.841 | 0.955 |
| Logistic Regression | SMOTE | 0.822 | 0.735 | 0.812 | 0.772 | 0.907 |
| Decision Tree | SMOTE | 0.847 | 0.764 | 0.849 | 0.804 | 0.932 |
| Random Forest | SMOTE | 0.868 | 0.824 | 0.819 | 0.821 | 0.943 |
| XGBoost | SMOTE | 0.880 | 0.840 | 0.835 | 0.837 | 0.953 |

**Lectura.** El balanceo (class_weight o SMOTE) **sube el recall** —se detectan más cancelaciones— a costa de **bajar la precisión**, mientras que el **ROC-AUC apenas cambia** (es independiente del umbral). Es decir, no hace al modelo "mejor" en capacidad de ordenar, pero sí desplaza el compromiso hacia detectar más positivos, útil si al hotel le cuesta más una cancelación no detectada que una falsa alarma.

