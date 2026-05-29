# Análisis Profundo del Repositorio `pontia-ml`

Este documento presenta un análisis profundo del repositorio, enfocado en la calidad del código, explicabilidad, detalles de Machine Learning, exploración de datos y decisiones de diseño tomadas.

## 1. Calidad del Código y Explicabilidad (Software Engineering)

### Puntos Fuertes
- **Arquitectura Limpia y Modular**: La separación en `src/ml_hotel_cancellations/` entre pipeline (`ml/`), API (`api/`), interfaz (`ui/`) y utilidades (`utils/`) es excelente. Sigue buenas prácticas de empaquetado de Python.
- **Evitar Fugas de Datos (Data Leakage)**: El uso intensivo de `sklearn.pipeline.Pipeline` y `ColumnTransformer` garantiza que el preprocesamiento se ajuste únicamente en el conjunto de entrenamiento, evitando filtraciones hacia el test y facilitando la inferencia en producción.
- **Documentación y Tipado**: El código cuenta con anotaciones de tipo (`typing`) modernas y *docstrings* muy detallados en español. El `informe_final.md` y `interpretabilidad.md` son excepcionalmente claros y honestos en las justificaciones.
- **Centralización de Configuración**: Todo se gobierna desde `config.py`, evitando valores mágicos ("magic numbers") dispersos en el código. Esto facilita el mantenimiento.

### Áreas de Mejora
- **Serialización del Modelo Keras**: En `model_trainer.py`, la serialización del modelo `KerasMLPClassifier` en los métodos mágicos `__getstate__` y `__setstate__` escribe directamente en disco usando un `TemporaryDirectory`. Aunque funcional, esto añade latencia de I/O de disco al hacer *pickle* del pipeline y un grado de fragilidad. Una alternativa óptima sería usar las APIs de `tf.keras.models.clone_model` y pasar los pesos en memoria a variables de Python en formato lista/arrays, o aprovechar la nueva API de serialización nativa a bytes si se usa Keras 3.
- **Gestión de Semillas (Random State)**: El uso estricto de `random_state = 42` es fantástico para reproducibilidad y para comparar, pero evaluar los modelos sobre una sola partición *holdout* puede esconder varianza inherente en la capacidad del modelo. Incorporar Validación Cruzada Repetida (Repeated K-Fold) para el reporte de métricas finales brindaría un intervalo de confianza sobre el rendimiento, más robusto estadísticamente.

## 2. Detalles de Machine Learning (Modelado y Métricas)

### Puntos Fuertes
- **Comparativa Justa**: Todos los modelos (desde Regresión Logística hasta Redes Neuronales) consumen exactamente la misma matriz transformada a través del `Pipeline` central, aislando y evaluando justamente la capacidad algorítmica de cada técnica.
- **Análisis de Balanceo de Clases**: La decisión de no forzar SMOTE o `class_weight` en el pipeline final basándose en que el ROC-AUC se mantiene estable, denota una profunda comprensión teórica de cómo funcionan las métricas independientes de umbral.

### Áreas de Mejora
- **Validación Temporal (Out-of-Time Validation)**: Como se identifica en `docs/informe_final.md`, el dataset tiene un orden cronológico claro (2015-2017). Utilizar un *Time Series Split* o dejar el último bloque de meses estrictamente para test, revelaría de verdad cómo degrada el modelo ante cambios de tendencia del usuario en el tiempo (*Concept Drift*), lo que es crítico en cualquier empresa de *hospitality*.
- **Métrica PR-AUC vs ROC-AUC**: Para clases desbalanceadas (37% cancelaciones vs 63% no canceladas), la curva *Precision-Recall* (y calcular el PR-AUC) suele ser bastante más informativa y punitiva que la curva ROC-AUC. Agregar PR-AUC a las métricas del `evaluator.py` proporcionaría una perspectiva un grado más exigente.
- **Calibración Óptima del Umbral**: Actualmente el umbral de decisión para `predict()` es estático en `0.5` (`config.DECISION_THRESHOLD`). Implementar calibración post-entrenamiento (por ejemplo con `CalibratedClassifierCV` o buscando empíricamente el umbral sobre validación) que optimice el *Expected Value* del negocio, pesando el coste asimétrico entre *False Positives* y *False Negatives*.

## 3. Exploración y Análisis de Datos (Decisiones de Feature Engineering)

### Puntos Fuertes
- **Supresión Crítica de Fugas**: Descartar `reservation_status` y `reservation_status_date` previene que el modelo "memorice" resultados a posteriori, una trampa común en novatos de ML.
- **Descarte de `arrival_date_year`**: Es una decisión madura y brillante. Usar años incompletos crearía sobreajuste en un árbol de decisión al no permitirle generalizar en absoluto a reservas de 2018 o posteriores.

### Áreas de Mejora
- **Codificación de Variables Temporales Cíclicas**: Elementos como `arrival_date_month`, `arrival_date_week_number` y `day_of_month` son estacionarios. Los modelos lineales (Logistic Regression) y las redes neuronales se beneficiarían inmensamente de *Codificación Cíclica* (transformación matemática usando senos y cosenos en 2 variables por rasgo temporal). Esto les enseña que el mes 12 (Diciembre) es cronológicamente vecino del mes 1 (Enero).
- **Tratamiento de Alta Cardinalidad en `agent` y `country`**: Limitar el `OneHotEncoder` a `max_categories=25` previene la maldición de la dimensionalidad, pero penaliza a grupos sub-representados fusionándolos en "infrequent".
  - *Recomendación Práctica*: Aplicar **Target Encoding** o *Mean Encoding* en su lugar, agrupando solo donde las muestras sean ínfimas. Esto reemplazaría cientos de variables categóricas con 1 o 2 variables numéricas increíblemente potentes (la propensión histórica a la cancelación de dicho agente o país).
- **Aprovechamiento de los Ausentes (Missings)**:
  - `company`: Fue descartada inteligentemente por un 94% de nulos. Sin embargo, en la hotelería, el propio hecho de que *falte* la empresa suele diferenciar netamente a turistas particulares de empleados corporativos. Crear una variable binaria *booleana* `is_corporate` (1 si la compañía no es nula, 0 si lo es) podría rescatar una gran señal predictiva sin añadir ruido.

## 4. Explicabilidad del Modelo (SHAP y Análisis Agnóstico)

### Puntos Fuertes
- **Integración Multidimensional Avanzada**: Combinar SHAP a nivel global (Beeswarm, Bar), SHAP local (Waterfall plots para predicciones directas en Streamlit) e Importancia por Permutación (`scikit-learn`) rompe la barrera de que XGBoost sea una "caja negra" inescrutable.

### Áreas de Mejora
- **Partial Dependence Plots (PDP) / ICE Plots**: SHAP nos indica qué magnitudes influyen, pero no describe tan fácilmente *cómo es la curva de influencia*. Añadir gráficos PDP para variables cruciales continuas (como `lead_time` o `adr`) mostraría la forma exacta de la relación a ojos de negocio: por ejemplo, averiguar visualmente si a partir de "X días de antelación" el riesgo asume una asíntota, lo cual informaría al hotel sobre cuándo bloquear la devolución del depósito.
- **Explicador Universal (KernelExplainer)**: Ya que actualmente se usa intensivamente `TreeExplainer`, SHAP queda fuertemente anclado a XGBoost/Random Forest. Experimentar con `KernelExplainer` para el modelo base y la red neuronal (aunque computacionalmente más costoso) ayudaría a validar empíricamente si la red neuronal aprendió "reglas lógicas" similares a los árboles de decisión en casos locales controvertidos, dándole una capa extra de auditoría.
