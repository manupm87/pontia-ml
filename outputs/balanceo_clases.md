# Balanceo de clases — comparación de estrategias

Desbalance del problema: ~37 % de cancelaciones. Métricas sobre el conjunto de test, con hiperparámetros base (para aislar el efecto del balanceo).

| modelo | estrategia | accuracy | precision | recall | f1 | roc_auc |
|---|---|---|---|---|---|---|
| Logistic Regression | baseline | 0.825 | 0.806 | 0.695 | 0.747 | 0.906 |
| Decision Tree | baseline | 0.850 | 0.819 | 0.764 | 0.791 | 0.930 |
| Random Forest | baseline | 0.857 | 0.884 | 0.708 | 0.786 | 0.940 |
| XGBoost | baseline | 0.880 | 0.857 | 0.813 | 0.834 | 0.952 |
| Logistic Regression | class_weight | 0.819 | 0.731 | 0.810 | 0.768 | 0.906 |
| Decision Tree | class_weight | 0.846 | 0.795 | 0.789 | 0.792 | 0.925 |
| Random Forest | class_weight | 0.863 | 0.803 | 0.834 | 0.818 | 0.940 |
| XGBoost | class_weight | 0.874 | 0.805 | 0.873 | 0.837 | 0.952 |
| Logistic Regression | SMOTE | 0.820 | 0.734 | 0.807 | 0.769 | 0.906 |
| Decision Tree | SMOTE | 0.841 | 0.775 | 0.805 | 0.790 | 0.927 |
| Random Forest | SMOTE | 0.864 | 0.818 | 0.815 | 0.817 | 0.940 |
| XGBoost | SMOTE | 0.877 | 0.837 | 0.830 | 0.834 | 0.950 |

**Lectura.** El balanceo (class_weight o SMOTE) **sube el recall** —se detectan más cancelaciones— a costa de **bajar la precisión**, mientras que el **ROC-AUC apenas cambia** (es independiente del umbral). Es decir, no hace al modelo "mejor" en capacidad de ordenar, pero sí desplaza el compromiso hacia detectar más positivos, útil si al hotel le cuesta más una cancelación no detectada que una falsa alarma.

