# Informe final — Predicción de cancelaciones de reservas hoteleras

**Máster en IA, Cloud Computing y DevOps · Módulo de Machine Learning y Deep Learning**

> 📖 Todos los términos técnicos que aparecen en este informe están explicados, en
> lenguaje sencillo, en el [**Glosario**](glosario.md). Si lees una palabra que no
> conoces, búscala allí.

---

## 1. Roles de la pareja

La práctica se ha realizado por parejas. El reparto de tareas ha sido el siguiente
(rellenar/ajustar con los datos reales del/de la segundo/a integrante):

| Integrante | Responsabilidades principales | Aportaciones concretas |
|------------|-------------------------------|------------------------|
| **Manuel Pérez** (manugijon@gmail.com) | Arquitectura del sistema, modelado y documentación | Diseño del paquete `src/`, proceso de entrenamiento (`train.py`), integración de la red neuronal, evaluación y redacción de README/informe/glosario |
| **[Nombre compañero/a]** (*[email]*) | *[p. ej.: exploración de datos, preprocesado y validación]* | *[p. ej.: notebook de análisis exploratorio, diseño del preprocesado, pruebas de modelos, revisión de resultados]* |

> La contribución individual es trazable mediante el **historial de commits** (el
> registro de cambios) del repositorio. Si no se distinguen roles, ambos integrantes
> reciben la misma nota (según el enunciado).

---

## 2. Justificación del problema

Las **cancelaciones de reservas** son un problema económico de primer orden en el
sector hotelero. Una cancelación tardía deja una habitación vacía que difícilmente
se vuelve a vender, distorsiona la previsión de ocupación y complica la gestión de
personal y recursos. La práctica habitual para mitigarlo —el *overbooking* (aceptar
más reservas de las plazas disponibles contando con que algunas se cancelarán)— solo
es segura si se puede **estimar el riesgo de cancelación** de cada reserva.

Por eso predecir `is_canceled` (la *variable objetivo*, lo que queremos adivinar) es
un caso de uso **realista y de alto valor**:

- **Aplicación directa:** priorizar reservas para *overbooking*, lanzar campañas de
  retención o pedir un depósito a las reservas de mayor riesgo.
- **Respuesta de dos valores (clasificación binaria):** cancelada (1) o no (0).
- **Datos ricos:** ~119 000 reservas y 31 *características* (columnas) de tipos muy
  distintos: temporales, categóricas (texto) y numéricas, ideales para ilustrar un
  preprocesado completo y comparar modelos.

---

## 3. Análisis exploratorio de datos (EDA)

El **EDA** (*Exploratory Data Analysis*) es la fase en la que exploramos los datos
con tablas y gráficos **antes de modelar**, para entenderlos y tomar decisiones con
fundamento. El análisis completo y reproducible está en
[`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb). Principales hallazgos y la
decisión de diseño que provocó cada uno:

### 3.1. La variable objetivo está desbalanceada

- **~37 % de las reservas se cancelan** (44 224 de 119 390); el 63 % no. A esto se
  le llama **desbalance de clases**.
- *Decisión:* usar una **partición estratificada** (mantener ese 37 % tanto en los
  datos de entrenamiento como en los de prueba) y elegir **ROC-AUC** como métrica
  principal en lugar de la *accuracy* (ver sección 5).

### 3.2. Hay columnas que "hacen trampa" (*fuga de información*)

La **fuga de información** (*data leakage*) ocurre cuando el modelo usa, sin querer,
datos que revelan la respuesta o que no existirían en el momento de predecir.

- `reservation_status` vale `Canceled`/`No-Show` **exactamente** cuando
  `is_canceled = 1`. Junto con `reservation_status_date`, describen lo que pasó
  **después** de decidir la cancelación.
- *Decisión:* eliminar ambas columnas siempre. Si no, el modelo "vería la
  respuesta" y obtendría un ~100 % de acierto **engañoso e inútil**.

### 3.3. Valores ausentes (huecos en los datos)

Un **valor ausente** (o nulo, *NaN*) es una celda vacía. La **imputación** es
rellenarla con un valor razonable.

| Columna | % de huecos | Qué hacemos |
|---------|:-----------:|-------------|
| `company` | ~94 % | **Eliminar** (apenas aporta información) |
| `agent` | ~14 % | Tratar como categoría; huecos → `"Unknown"` (desconocido) |
| `country` | ~0.4 % | Imputar con la constante `"Unknown"` |
| `children` | residual | Imputar con la **mediana** (el valor central) |

### 3.4. Variables numéricas

- `lead_time` (días de antelación entre la reserva y la llegada) es la numérica que
  **más se relaciona** con la cancelación: a más antelación, más probabilidad de
  cancelar.
- `total_of_special_requests` y `required_car_parking_spaces` se relacionan al
  revés (clientes más comprometidos cancelan menos).
- Como las variables están en escalas muy diferentes, aplicamos
  **estandarización** (ponerlas todas en una escala comparable, media 0 y
  desviación 1).

**Descartamos `arrival_date_year`.** El dataset tiene cuatro variables de fecha de
llegada (`year`, `month`, `week_number`, `day_of_month`). Las tres últimas aportan
**estacionalidad** (épocas del año en que se cancela más o menos), pero el **año**
no aporta valor real, por tres motivos:

- **Apenas discrimina:** la tasa de cancelación es casi idéntica los tres años
  (2015: 37.0 %, 2016: 35.9 %, 2017: 38.7 %).
- **No generaliza:** el modelo debe puntuar reservas *futuras*. Un año que no vio al
  entrenar (2018 en adelante) no tiene un valor de "año" interpretable: los árboles
  lo meterían en el último tramo conocido y un modelo lineal extrapolaría una
  tendencia inexistente.
- **Va confundido con la estación:** el dataset cubre **años parciales** (2015 solo
  jul–dic; 2017 solo ene–ago), de ahí que el año tenga una correlación de **−0.54**
  con `week_number` (la más alta de todas las numéricas). Esa señal estacional ya la
  recogen `month` y `week_number`, que **sí se repiten** cada año.

*Decisión:* eliminar `arrival_date_year`. En una partición aleatoria, incluirla
subía el ROC-AUC de XGBoost en ~0.003, pero es una mejora **engañosa** (optimismo
que no se trasladaría a producción, donde siempre se predice el futuro). Renunciar a
ella hace el modelo más honesto, en línea con la limitación de *validación temporal*
de la §7.

### 3.5. Variables categóricas (de texto)

- `deposit_type = "Non Refund"` (depósito no reembolsable) tiene una tasa de
  cancelación **cercana al 99 %**: una variable muy predictiva.
- El *City Hotel* cancela más que el *Resort Hotel*.
- Algunas categóricas tienen **muchísimos valores distintos** (*alta cardinalidad*):
  `country` (178 países) y `agent` (334 agencias). Para convertirlas en números sin
  crear cientos de columnas, usamos **codificación one-hot con un límite de
  categorías** (las menos frecuentes se agrupan en una sola).

### 3.6. Limpieza adicional

- Eliminamos ~180 reservas **sin ningún huésped** (`adults+children+babies = 0`),
  que son registros claramente erróneos.

---

## 4. Diseño del sistema

El proyecto está construido como un **paquete de software modular** (la carpeta
`src/`), separando cada responsabilidad en un fichero, en lugar de amontonar todo en
un único notebook.

### 4.1. El flujo de trabajo (*pipeline*)

Un **pipeline** ("tubería") es una secuencia de pasos encadenados. El nuestro va de
los datos crudos hasta el modelo elegido:

```text
 CSV crudo ──► data_loader ──► preprocessing ──► model_trainer ──► evaluator ──► best_model.pkl
              (cargar +        (preparar los     (entrenar los     (medir y
               limpiar +        datos: rellenar    5 modelos)        comparar:
               dividir en       huecos, escalar                      ROC, matriz de
               train/test)      y codificar)                         confusión...)
```

### 4.2. Para qué sirve cada módulo

| Módulo (fichero) | Responsabilidad |
|------------------|-----------------|
| `config.py` | Punto único de configuración: rutas, semilla aleatoria, listas de columnas, ajustes (*hiperparámetros*) de los modelos y métrica principal. |
| `data_loader.py` | Cargar el CSV, marcar los huecos, eliminar las columnas que "hacen trampa" y `company`, limpiar y **dividir** en entrenamiento/prueba de forma estratificada. |
| `preprocessing.py` | Construir el preprocesador: **imputar + estandarizar** las numéricas y **codificar (one-hot)** las categóricas. |
| `model_trainer.py` | Clase `ModelTrainer` que envuelve cada modelo en un `Pipeline` (preprocesado + modelo) y los entrena; incluye un envoltorio para que la red neuronal de Keras se comporte como el resto. |
| `evaluator.py` | Clase `Evaluator`: calcula las métricas, monta la tabla comparativa, elige el mejor modelo y dibuja los gráficos. |
| `train.py` | **Programa principal** que ejecuta todo el flujo y guarda los resultados. |
| `predict.py` | Hacer predicciones con `best_model.pkl` sobre reservas nuevas. |

### 4.3. Decisiones de ingeniería destacables (y por qué)

- **Uso de `Pipeline`:** encadenar preprocesado + modelo en un solo objeto tiene una
  ventaja clave: el preprocesado **aprende solo de los datos de entrenamiento**, lo
  que evita *fugas de información* hacia los datos de prueba. Además, modelo y
  preprocesado se guardan juntos y la predicción es directa.
- **Comparación justa:** los cinco modelos comparten **exactamente el mismo
  preprocesado**.
- **Red neuronal integrada:** creamos un envoltorio (`KerasMLPClassifier`) para que
  la red de Keras tenga la misma interfaz (`fit`/`predict`) que los modelos de
  scikit-learn y se pueda **guardar y reutilizar** igual que ellos.
- **Reproducibilidad:** fijamos la **semilla aleatoria** (`random_state = 42`) en
  las divisiones y en los modelos para que los resultados se puedan repetir.

### 4.4. Los cinco modelos que comparamos

Cada uno representa una "familia" distinta de algoritmos (todos explicados en el
[glosario](glosario.md)):

1. **Regresión logística** — modelo lineal sencillo; sirve de **línea base**
   (referencia mínima a superar).
2. **Árbol de decisión** — una serie de preguntas tipo "sí/no"; fácil de entender.
3. **Random Forest ("bosque aleatorio")** — combina **muchos árboles** y promedia
   sus votos (técnica llamada *ensemble*).
4. **XGBoost** — variante muy eficiente de *gradient boosting*: añade árboles que
   van **corrigiendo los errores** de los anteriores.
5. **Red neuronal multicapa (MLP, con Keras/TensorFlow)** — capas de "neuronas"
   `64-32-16`, con *dropout* 0.3 (apaga neuronas al azar para no sobreajustar),
   activación ReLU, salida *sigmoide* (da una probabilidad) y **early stopping**
   (para de entrenar cuando deja de mejorar).

### 4.5. Herramientas de producción y su equivalente en clase (`recursos/`)

El paquete `src/` está pensado como **sistema de producción**, así que en varios
puntos usa utilidades de scikit-learn más robustas que las vistas en los notebooks
de clase (`recursos/`). Cada una **hace lo mismo** que su equivalente de clase,
pero de forma reproducible y segura para la inferencia. Esta tabla las mapea (es la
documentación explícita de por qué `src/` se aparta de `recursos/`):

| Herramienta en `src/` (producción) | Equivalente en `recursos/` | Qué añade la versión de producción |
|---|---|---|
| `Pipeline(preprocessor, modelo)` | entrenar el modelo directamente sobre el DataFrame ya preparado a mano | Empaqueta preprocesado + modelo en **un único objeto**: el preprocesado se **aprende solo del train** (sin fugas) y se persiste junto al modelo para predecir. |
| `ColumnTransformer` | preparar cada grupo de columnas por separado con pandas | Aplica transformaciones distintas a numéricas y categóricas de forma declarativa dentro del `Pipeline`. |
| `OneHotEncoder(handle_unknown=…, max_categories=25)` | `pd.get_dummies(X)` | Recuerda las categorías del *train*, **tolera categorías no vistas** al predecir y **limita la cardinalidad** (evita cientos de columnas). |
| `SimpleImputer(strategy=…)` | `.fillna()` / `.dropna()` | Aprende el valor de relleno (p. ej. la mediana) **en train** y lo reaplica idéntico en test/predicción. |
| `StandardScaler` (todas las numéricas) | `StandardScaler` (solo KNN/SVM en clase) | Es la **misma** herramienta; en producción se aplica a todas las numéricas dentro del `Pipeline` por consistencia. |
| `RandomizedSearchCV` (Random Forest, XGBoost) | `GridSearchCV` | Muestrea combinaciones al azar cuando el espacio es grande, donde una búsqueda exhaustiva sería inviable. En los espacios pequeños (regresión logística, árbol) sí usamos `GridSearchCV`, **igual que en clase**. |

> El notebook `notebooks/playground/` replica el flujo **solo con las herramientas
> de `recursos/`** (`pd.get_dummies`, `GridSearchCV`, etc.), de modo que sirve de
> puente entre la versión "de clase" y la versión "de producción" de `src/`.

---

## 5. Resultados y elección final

Evaluamos sobre el **conjunto de prueba** (*test*): 23 842 reservas (20 % del total)
que el modelo **no usó al entrenar**, para medir si generaliza a casos nuevos.

| Modelo | Accuracy | Precision | Recall | F1 | **ROC-AUC** |
|--------|:--------:|:---------:|:------:|:--:|:-----------:|
| **XGBoost** ⭐ | 0.8934 | 0.8701 | 0.8374 | 0.8535 | **0.9614** |
| Red neuronal (Keras) | 0.8718 | 0.8427 | 0.8045 | 0.8231 | 0.9460 |
| Random Forest | 0.8644 | 0.8828 | 0.7313 | 0.8000 | 0.9455 |
| Árbol de decisión | 0.8551 | 0.8191 | 0.7819 | 0.8000 | 0.9329 |
| Regresión logística | 0.8190 | 0.7312 | 0.8093 | 0.7683 | 0.9064 |

> Estas cifras se obtienen con los **hiperparámetros optimizados** por validación
> cruzada (ver §6, bonus), que el pipeline usa por defecto.

> **Recordatorio de métricas** (detalle en el [glosario](glosario.md)):
> *accuracy* = % de aciertos · *precision* = pocas falsas alarmas · *recall* = se
> escapan pocas cancelaciones · *F1* = equilibrio de las dos · *ROC-AUC* = capacidad
> de ordenar bien por riesgo (0.5 = azar, 1 = perfecto).

![Curva ROC comparativa](../outputs/roc_curves.png)

*La curva ROC enfrenta cancelaciones detectadas (eje Y) frente a falsas alarmas (eje
X) según el umbral; cuanto más cerca de la esquina superior izquierda, mejor.*

![Matrices de confusión](../outputs/confusion_matrices.png)

*La matriz de confusión cruza lo predicho con lo real: los aciertos están en la
diagonal.*

![Importancia de variables](../outputs/feature_importance.png)

*La importancia de variables indica qué características influyen más en las
predicciones del Random Forest.*

### 5.1. Elección del modelo y de la métrica

- **Métrica principal: ROC-AUC.** Es robusta al desbalance, no depende del umbral de
  decisión (lo que permite ajustar la "agresividad" del *overbooking*) y es
  comparable entre modelos. Reportamos además *recall* y *F1* por su lectura de
  negocio.
- **Modelo elegido: XGBoost** (ROC-AUC = 0.961). Supera al resto en la métrica
  principal y en F1, y aun así entrena en pocos segundos (~3.6 s). Se guarda como
  `models/best_model.pkl`.

### 5.2. Qué significan estos resultados para el hotel

- XGBoost detecta el **84 % de las cancelaciones reales** (*recall* 0.84) con una
  **precisión del 87 %**: un buen equilibrio para actuar sin generar demasiadas
  falsas alarmas.
- El Random Forest es el más **conservador** (más precisión pero menos recall):
  preferible si una falsa alarma fuese muy costosa.

---

## 6. Bonus técnicos implementados

Más allá de los requisitos mínimos, añadimos los siguientes extras (el enunciado
los puntúa como *bonus*).

### 6.1. Optimización de hiperparámetros

Buscamos automáticamente la mejor configuración de cada modelo clásico mediante
**validación cruzada** (3 particiones), optimizando ROC-AUC:

- **GridSearchCV** (búsqueda exhaustiva) para los espacios pequeños: regresión
  logística y árbol de decisión.
- **RandomizedSearchCV** (muestreo aleatorio) para los grandes: Random Forest y
  XGBoost.

Los mejores hiperparámetros se **persisten** en `outputs/best_hiperparametros.json`
y el pipeline los **usa por defecto** (`python -m src.train`); rehacer la búsqueda
es tan simple como `python -m src.train --tune` o `python -m src.tuning`. Partiendo
de unos valores base ya buenos hallados explorando a mano
(`max_depth=14, n_estimators=500, learning_rate=0.1`; **0.9573** de ROC-AUC en
validación cruzada), el finetuning encontró
`max_depth=16, n_estimators=600, learning_rate=0.03`, subiendo el ROC-AUC de CV a
**0.9586** y alcanzando **0.9614 en test** (el mejor resultado del proyecto). El
detalle (CV base vs. optimizada) queda en `outputs/tuning_hiperparametros.md`.
Implementado en `src/tuning.py`.

> *Nota de hardware:* el código es **GPU-aware** — XGBoost puede entrenar en una
> GPU NVIDIA con `PONTIA_USE_GPU=1` —, pero por defecto se ejecuta en **CPU**,
> porque a esta escala de datos la GPU no acelera y la CPU es plenamente
> reproducible (`src/gpu.py`).

### 6.2. Balanceo de clases

El problema está moderadamente desbalanceado (~37 % de cancelaciones). Comparamos
tres estrategias (`src/balancing.py`, resultados en `outputs/balanceo_clases.md`
y `.png`): **sin balanceo**, **class_weight** (reponderar la clase minoritaria;
`scale_pos_weight` en XGBoost) y **SMOTE** (sobremuestreo sintético con
*imbalanced-learn*, aplicado solo al entrenamiento).

| XGBoost | recall | precision | ROC-AUC |
|---|:--:|:--:|:--:|
| Sin balanceo | 0.81 | 0.86 | 0.952 |
| class_weight | **0.87** | 0.81 | 0.952 |
| SMOTE | 0.83 | 0.84 | 0.950 |

**Conclusión:** el balanceo **sube el recall** (detecta más cancelaciones) a costa
de **precisión**, y el **ROC-AUC apenas cambia** (es independiente del umbral). Por
eso el pipeline principal **no** balancea: como optimizamos ROC-AUC, el compromiso
recall/precisión se ajusta mejor **moviendo el umbral** de decisión según el coste
de negocio (una cancelación no detectada vs. una falsa alarma).

---

## 7. Reflexión crítica: limitaciones y mejoras

Ser honestos con las limitaciones forma parte de un buen trabajo de ML.

**Limitaciones actuales**

- **Validación temporal pendiente:** dividimos los datos al azar. Como las reservas
  tienen fecha (2015–2017), una división **por tiempo** (entrenar con el pasado y
  probar con el futuro) sería más realista y probablemente daría una cifra algo más
  baja pero más fiable.
- **Desbalance sin tratamiento explícito:** lo abordamos con estratificación y una
  métrica adecuada, pero no con técnicas específicas de reequilibrado.
- **Alta cardinalidad simplificada:** al limitar las categorías de `country`/`agent`
  perdemos parte de la información de las menos frecuentes.
- **Umbral fijo en 0.5:** no lo hemos ajustado a un objetivo concreto de negocio.

**Líneas de mejora (trabajo futuro)**

- **Validación temporal** (entrenar con el pasado y probar con el futuro) para
  cifras más fiables que la división aleatoria actual.
- **Reequilibrado de clases** (p. ej. `class_weight`, SMOTE).
- **Interpretabilidad avanzada** con SHAP (explica cada predicción individual).
- **Calibración de probabilidades** y ajuste del umbral según coste/beneficio.
- **Productivización:** exponer el modelo mediante una API web y registrar los
  experimentos (p. ej. con MLflow).

---

> **Reproducibilidad.** Todos los resultados de este informe se generan ejecutando
> `python -m src.train` desde la carpeta `project/`, con las librerías de
> `requirements.txt` (Python 3.12). Las tablas y figuras provienen de `outputs/`.
