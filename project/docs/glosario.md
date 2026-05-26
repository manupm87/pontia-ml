# 📖 Glosario — explicación de todos los términos

Este documento explica, **en lenguaje sencillo**, todos los conceptos, términos
técnicos y siglas que aparecen en el proyecto. Está pensado para que cualquier
persona pueda entender el código y la documentación aunque no tenga experiencia
previa en Machine Learning.

> 💡 A lo largo del README y del informe encontrarás enlaces a este glosario.

---

## 1. Conceptos básicos

**IA (Inteligencia Artificial).** Campo que busca que las máquinas realicen tareas
que asociamos a la inteligencia humana (decidir, reconocer, predecir...).

**ML (Machine Learning, "aprendizaje automático").** Rama de la IA en la que el
programa **aprende patrones a partir de ejemplos** (datos) en lugar de seguir
reglas escritas a mano. Aquí aprende a distinguir reservas que se cancelan de las
que no.

**DL (Deep Learning, "aprendizaje profundo").** Subconjunto del ML que usa **redes
neuronales** con muchas capas. Lo usamos para la red neuronal del proyecto.

**Aprendizaje supervisado.** Tipo de ML en el que entrenamos con ejemplos
**etiquetados**: para cada reserva del pasado ya sabemos si se canceló o no, y el
modelo aprende de esas respuestas conocidas.

**Clasificación binaria.** Problema en el que la respuesta solo puede tomar **dos
valores**. Aquí: cancelada (1) o no cancelada (0).

**Variable objetivo (target).** La columna que queremos **predecir**. En este
proyecto es `is_canceled`.

**Característica / variable predictora (feature).** Cada columna de entrada que el
modelo usa para predecir (p. ej. `lead_time`, `country`, `adr`).

**Modelo.** El "programa que ha aprendido". Recibe las características de una reserva
y devuelve una predicción.

**Entrenar (fit).** Proceso de mostrarle al modelo los datos para que aprenda los
patrones.

**Inferencia / predicción (predict).** Usar un modelo ya entrenado para obtener la
respuesta sobre datos nuevos.

**Parámetro vs hiperparámetro.**
- *Parámetros*: los valores internos que el modelo **aprende solo** al entrenar.
- *Hiperparámetros*: los ajustes que **decidimos nosotros antes** de entrenar
  (p. ej. cuántos árboles tiene un Random Forest). En el proyecto están en
  `src/config.py`.

**Sobreajuste (overfitting).** Cuando el modelo "se memoriza" los datos de
entrenamiento y luego falla con datos nuevos. Es el principal enemigo a vigilar.

**Infraajuste (underfitting).** Lo contrario: el modelo es demasiado simple y ni
siquiera aprende bien los datos de entrenamiento.

**Reproducibilidad / semilla aleatoria (`random_state`).** Muchos algoritmos usan
azar (p. ej. al barajar datos). Fijar una "semilla" (un número, aquí el 42) hace
que el azar sea **siempre el mismo**, de forma que los resultados se puedan repetir.

---

## 2. Datos y preprocesado

**Preprocesado / preprocesamiento.** Conjunto de transformaciones que preparan los
datos crudos para que el modelo pueda usarlos (limpiar, rellenar huecos, poner todo
en una escala parecida, convertir texto en números...).

**Conjunto de entrenamiento y de prueba (train / test).** Dividimos los datos en
dos partes: una para **entrenar** (el modelo aprende) y otra, que no ha visto, para
**evaluar** cómo de bien generaliza. Aquí usamos 80 % / 20 %.

**Partición estratificada.** Forma de dividir los datos que **mantiene la misma
proporción de clases** en train y test (≈37 % de cancelaciones en ambos). Importante
cuando las clases están desbalanceadas.

**Validación cruzada (cross-validation, CV).** Técnica para evaluar de forma más
fiable: se reparten los datos en *k* bloques y se entrena/evalúa *k* veces rotando
el bloque de prueba. Da una media y una variabilidad de la métrica.

**Valor ausente / nulo (NaN).** Un dato que falta (celda vacía). "NaN" significa
*Not a Number*. En el CSV original aparecen como la palabra `NULL`.

**Imputación.** Rellenar los valores ausentes con un valor razonable. Aquí: la
**mediana** (el valor central) para las numéricas y la categoría `"Unknown"`
(desconocido) para las categóricas.

**Variable numérica vs categórica.**
- *Numérica*: número con el que se puede operar (`lead_time`, `adr`).
- *Categórica*: etiqueta de un conjunto cerrado de opciones (`hotel`, `country`).

**Escalado / estandarización (StandardScaler).** Poner las variables numéricas en
una **escala comparable** (media 0 y desviación 1). Evita que una variable con
números grandes (como `adr`) "pese" más que otra solo por su tamaño. Es importante
para la regresión logística y la red neuronal.

**Codificación one-hot (One-Hot Encoding).** Convierte una variable categórica en
varias columnas de 0/1, una por categoría. Ejemplo: `hotel` ("City"/"Resort") se
convierte en dos columnas `hotel_City` y `hotel_Resort`. Necesario porque los
modelos solo entienden números.

**Cardinalidad.** Número de valores distintos de una variable categórica. `country`
tiene **alta cardinalidad** (178 países). Si hiciéramos one-hot de todos, saldrían
cientos de columnas, así que limitamos el número de categorías
(`max_categories`) y agrupamos las raras en una categoría "poco frecuente".

**Desbalance de clases.** Cuando una clase es mucho más frecuente que la otra. Aquí
hay más reservas no canceladas (63 %) que canceladas (37 %): un desbalance moderado.

**Fuga de información (data leakage).** Error grave en el que el modelo usa, sin
querer, información que **no estaría disponible en el momento de predecir** o que
**revela la respuesta**. Aquí `reservation_status` ("Canceled"/"Check-Out") dice
directamente si se canceló → hay que eliminarla, o el modelo "haría trampa".

**Pipeline ("tubería").** Objeto que **encadena varios pasos** (preprocesado +
modelo) en una sola unidad. Ventaja: el preprocesado se aprende **solo con los datos
de entrenamiento**, evitando fugas, y se guarda junto al modelo.

**ColumnTransformer ("transformador por columnas").** Herramienta de scikit-learn
que aplica **transformaciones distintas a distintas columnas** (escalar las
numéricas y codificar las categóricas) dentro del mismo Pipeline.

---

## 3. Los modelos que comparamos

**Regresión logística (Logistic Regression).** Modelo **lineal** sencillo: estima
la probabilidad de cancelación combinando las variables con unos pesos. Sirve como
**línea base** (referencia mínima a superar).

**Árbol de decisión (Decision Tree).** Modelo con forma de árbol de preguntas
("¿lead_time > 30?", "¿depósito no reembolsable?"...) que va dividiendo los datos
hasta decidir. Fácil de interpretar pero tiende al sobreajuste.

**Ensemble ("conjunto").** Técnica que **combina muchos modelos** para obtener uno
más robusto que cada uno por separado.

**Bagging.** Tipo de ensemble que entrena muchos modelos en paralelo sobre muestras
distintas de los datos y promedia sus respuestas.

**Random Forest ("bosque aleatorio").** Ensemble de **muchos árboles de decisión**
(bagging). Cada árbol vota y se promedia. Más preciso y estable que un solo árbol.

**Boosting.** Tipo de ensemble que entrena modelos **de forma secuencial**, donde
cada nuevo modelo se centra en corregir los errores del anterior.

**Gradient Boosting.** Boosting que añade árboles para reducir progresivamente el
error, guiándose por el "gradiente" (la dirección de mejora) de una función de error.

**XGBoost (eXtreme Gradient Boosting).** Una implementación muy eficiente y popular
de gradient boosting. Suele dar muy buenos resultados con datos en tablas. **Es el
mejor modelo de este proyecto.**

**Red neuronal (neural network).** Modelo inspirado en el cerebro: capas de
"neuronas" conectadas que transforman la entrada poco a poco hasta dar una salida.

**MLP (Multi-Layer Perceptron, "perceptrón multicapa").** El tipo de red neuronal
más básico: capas densas una detrás de otra. Es la red que usamos.

Términos de la red neuronal:
- **Neurona / capa.** Unidad de cálculo / grupo de neuronas al mismo nivel.
- **Capa densa (Dense).** Capa en la que cada neurona se conecta con todas las de
  la capa anterior.
- **Función de activación.** Función que introduce "no linealidad" para que la red
  aprenda patrones complejos.
  - **ReLU.** Activación habitual en capas internas: deja pasar los positivos y
    pone a 0 los negativos.
  - **Sigmoide.** Activación de la capa final: convierte el resultado en una
    **probabilidad entre 0 y 1**.
- **Dropout.** Técnica de regularización: "apaga" al azar un porcentaje de neuronas
  en cada paso de entrenamiento para evitar el sobreajuste.
- **Optimizador (Adam).** Algoritmo que ajusta los pesos de la red para reducir el
  error. *Adam* es uno de los más usados y eficientes.
- **Función de pérdida (loss).** Mide cómo de equivocada va la red; el entrenamiento
  intenta minimizarla. Para clasificación binaria usamos *binary crossentropy*
  (entropía cruzada binaria).
- **Época (epoch).** Una pasada completa por todos los datos de entrenamiento.
- **Lote (batch).** Grupo de ejemplos que la red procesa a la vez antes de
  actualizar sus pesos.
- **Early stopping ("parada temprana").** Detiene el entrenamiento cuando el modelo
  deja de mejorar en datos de validación, evitando sobreajustar y ahorrando tiempo.
- **Validation split.** Porción de los datos de entrenamiento reservada para vigilar
  el rendimiento durante el entrenamiento (no para entrenar).

---

## 4. Cómo medimos los modelos (métricas)

**Matriz de confusión.** Tabla que cruza lo predicho con lo real. Sus cuatro
casillas:
- **VP / TP (Verdadero Positivo).** Predijo cancela y **sí** canceló. ✅
- **VN / TN (Verdadero Negativo).** Predijo no cancela y **no** canceló. ✅
- **FP (Falso Positivo).** Predijo cancela pero **no** canceló (falsa alarma). ❌
- **FN (Falso Negativo).** Predijo no cancela pero **sí** canceló (se nos escapó). ❌

**Accuracy (exactitud).** Porcentaje de aciertos totales. Engaña cuando hay
desbalance (si el 63 % no cancela, decir "nunca cancela" ya acierta el 63 %).

**Precision (precisión).** De todas las que predije como cancelaciones, ¿cuántas lo
eran de verdad? `TP / (TP + FP)`. Alta precisión = pocas falsas alarmas.

**Recall (sensibilidad o exhaustividad).** De todas las cancelaciones reales,
¿cuántas detecté? `TP / (TP + FN)`. Alto recall = se escapan pocas cancelaciones.

**F1-score.** Media equilibrada (media armónica) entre precisión y recall. Útil
cuando queremos un balance entre ambas y hay desbalance de clases.

**Curva ROC (Receiver Operating Characteristic).** Gráfico que muestra el equilibrio
entre detectar positivos y generar falsas alarmas a medida que cambiamos el umbral
de decisión. Enfrenta:
- **TPR (True Positive Rate).** = recall (positivos detectados).
- **FPR (False Positive Rate).** Negativos clasificados por error como positivos.

**AUC (Area Under the Curve, "área bajo la curva").** Número entre 0.5 (azar) y 1
(perfecto) que **resume la curva ROC** en un solo valor.

**ROC-AUC.** El AUC de la curva ROC. **Es nuestra métrica principal** porque:
es robusta al desbalance, no depende del umbral elegido y permite comparar modelos
muy distintos. Mide la capacidad del modelo de **ordenar** bien las reservas por
riesgo de cancelación.

**Probabilidad y umbral de decisión.** Los modelos no dicen "cancela/no cancela"
directamente: dan una **probabilidad** (p. ej. 0.73). El **umbral** (por defecto
0.5) es el corte a partir del cual decidimos que es una cancelación.

**Importancia de variables (feature importances).** Medida de cuánto influye cada
característica en las decisiones del modelo. La dibujamos para el Random Forest y
ayuda a **interpretar** qué factores predicen mejor la cancelación.

---

## 5. Herramientas, librerías y formatos

**Python.** Lenguaje de programación usado en todo el proyecto.

**Entorno virtual (venv).** "Caja" aislada donde se instalan las librerías de **este**
proyecto, sin mezclarlas con las del resto del sistema. Garantiza que todos usen las
mismas versiones.

**Librería / paquete.** Conjunto de código ya hecho que reutilizamos.

**CSV (Comma-Separated Values).** Fichero de texto con datos en forma de tabla,
separados por comas. Es el formato del dataset.

**Dataset.** El conjunto de datos con el que trabajamos (las ~119 000 reservas).

**pandas.** Librería para manejar tablas de datos (DataFrames) en Python.

**NumPy.** Librería para cálculo numérico y operaciones con vectores y matrices.

**scikit-learn (sklearn).** Librería principal de Machine Learning clásico:
preprocesado, modelos, métricas, Pipelines...

**XGBoost.** Librería del modelo del mismo nombre (ver sección 3).

**TensorFlow / Keras.** TensorFlow es la librería de Deep Learning de Google; *Keras*
es su interfaz sencilla para construir redes neuronales. Aquí se usan para el MLP.

**matplotlib / seaborn / plotly.** Librerías para crear gráficos (estáticos las dos
primeras, interactivos plotly).

**Jupyter Notebook (`.ipynb`).** Documento interactivo que mezcla texto, código y
resultados (gráficos, tablas). Lo usamos para el análisis exploratorio y la
comparativa.

**EDA (Exploratory Data Analysis, "análisis exploratorio de datos").** Fase inicial
en la que exploramos los datos con tablas y gráficos para **entenderlos** y tomar
decisiones de modelado.

**Serialización / joblib / pickle / `.pkl`.** "Serializar" es guardar un objeto de
Python (aquí, un modelo entrenado) en un fichero para reutilizarlo después. `joblib`
es la herramienta que usamos y `.pkl` la extensión del fichero resultante.

**ADR (Average Daily Rate).** Término del sector hotelero: **precio medio por noche**
de la reserva (columna `adr`).
