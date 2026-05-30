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
  `src/ml_hotel_cancellations/config.py`.

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
tiene **alta cardinalidad** (~178 países), igual que `agent` y `company`. Si
hiciéramos one-hot de todas, saldrían cientos de columnas, así que aplicamos
**reducción de cardinalidad supervisada** (ver entrada propia): conservamos las
categorías con soporte y señal fuerte y agrupamos el resto en `"Otros"`.

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

**Fit-on-train ("ajustar solo con entrenamiento").** Aprender los parámetros de
cualquier transformación (medianas, escalas, categorías frecuentes...) usando **solo
el conjunto de entrenamiento**, y aplicarla luego al test. Es la forma de evitar la
fuga de información (data leakage).

**Ausencia informativa.** Cuando el hecho de que un valor **falte ES información**, no
ruido. Aquí, que una reserva no tenga `company` o `agent` asociados se relaciona con un
riesgo de cancelación distinto, así que en lugar de tirar el dato lo convertimos en una
feature binaria (`has_company`, `has_agent`).

**Feature derivada.** Variable **nueva** calculada a partir de las originales (aquí
`has_company`, `has_agent` y `noches`, sumando estancias entre semana y fin de semana).

**Reducción de cardinalidad (supervisada).** Agrupar las categorías **raras** de una
variable de alta cardinalidad (`agent`, `country`, `company`, con cientos de valores)
en una categoría "Otros", conservando solo las que tienen suficiente soporte y una
señal fuerte respecto al target. Se ajusta **solo con train** (*fit-on-train*) para no
filtrar información del test.

**SMOTE.** Técnica que genera ejemplos sintéticos de la clase minoritaria para
equilibrar el conjunto de entrenamiento. En este proyecto **no forma parte del paquete
de producción**: se exploró en el notebook playground 06.

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

**SHAP (SHapley Additive exPlanations).** Técnica de **interpretabilidad** que
reparte cada predicción entre las variables: dice cuánto ha empujado cada una hacia
"cancela" o "no cancela". Se basa en los *valores de Shapley* (teoría de juegos).
Permite una lectura **global** (qué pesa más en todo el conjunto) y **local** (por
qué una reserva concreta se predice así). Ver `docs/interpretabilidad.md`.

**Importancia por permutación.** Forma sencilla y *agnóstica al modelo* de medir
importancia: se baraja al azar una variable y se mira cuánto empeora la métrica; si
empeora mucho, esa variable era importante. Sirve para cualquier modelo (no solo
árboles).

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

**API REST.** Una *API* (Interfaz de Programación de Aplicaciones) es una "puerta de
entrada" para que **otros programas** usen nuestro modelo sin conocer su código:
envían datos y reciben una respuesta. *REST* es el estilo más común para APIs web
(usa peticiones HTTP como `GET`/`POST`). Aquí exponemos la predicción por una API.

**Endpoint.** Cada "dirección" concreta de la API que hace una cosa (p. ej.
`POST /predict` para predecir, `GET /health` para comprobar el estado).

**FastAPI / Uvicorn.** *FastAPI* es la librería con la que construimos la API en
Python (rápida y con validación automática de los datos de entrada). *Uvicorn* es el
servidor que la ejecuta (`uvicorn ml_hotel_cancellations.api.main:app`).

**Swagger / OpenAPI.** Documentación **interactiva** de la API que FastAPI genera
sola en `/docs`: permite ver los endpoints y probarlos desde el navegador.

**Streamlit.** Librería para crear **aplicaciones web** de datos en Python con muy
poco código. La usamos para la interfaz visual (`src/ml_hotel_cancellations/ui/app.py`): tablas, gráficos y un
formulario de predicción que llama a la API.

**MLOps.** Conjunto de prácticas que aplican la disciplina del *DevOps* (despliegue
continuo, automatización, monitorización) al ciclo de vida de los modelos de
*Machine Learning*: entrenamiento, registro, despliegue, monitorización y
re-entrenamiento. Su objetivo es que un modelo pase de un *notebook* a un servicio
en producción de forma trazable y repetible.

**MLflow.** Plataforma de código abierto de MLOps con dos piezas centrales que
usa este proyecto: **MLflow Tracking** registra cada ejecución de entrenamiento
(hiperparámetros, métricas, artefactos) en un servidor central, y **MLflow Model
Registry** versiona los modelos resultantes y les asocia *stages* (`Staging`,
`Production`, `Archived`).

**Run / Experiment (MLflow).** Un *run* es una ejecución concreta de entrenamiento;
un *experiment* agrupa runs relacionados. El proyecto crea un *run* padre por cada
ejecución de los scripts `train` y `tune`, con uno o
varios *child runs* anidados (uno por modelo o por combinación).

**Stage / Promoción de un modelo.** Cada versión registrada en el *Model Registry*
puede transicionarse entre cuatro *stages* (`None`, `Staging`, `Production`,
`Archived`). El alias `models:/<nombre>/Production` siempre apunta a la versión más
reciente en ese *stage*: promocionar una v2 hace que cualquier código que use ese
alias pase automáticamente a servirla.

**DagsHub.** Plataforma gratuita y pública que ofrece un servidor MLflow gestionado
para repositorios de GitHub. Permite registrar experimentos sin auto-hospedar
infraestructura. En este proyecto, la URL `https://dagshub.com/<usuario>/<repo>.mlflow`
expone el servidor de tracking.

**Render.** Servicio de hosting que despliega aplicaciones web a partir de un
repositorio Git. El *tier* gratuito da 512 MB de RAM por servicio y suspende el
contenedor tras 15 min de inactividad, lo que provoca una latencia de arranque en
frío de 30-50 s en la siguiente petición. En este proyecto aloja la API FastAPI.

**Streamlit Community Cloud.** Servicio de hosting específico para aplicaciones
Streamlit, con despliegue directo desde GitHub. Tiene aproximadamente 1 GB de RAM
compartida y se reactiva en cuestión de segundos. En este proyecto aloja la
interfaz visual.

**Cold start / Warm restart.** *Cold start* es la latencia adicional que se observa
cuando un contenedor que estaba apagado tiene que arrancar para atender una
petición. *Warm restart* es un reinicio del proceso que conserva el sistema de
ficheros (en particular, los modelos previamente descargados a `/tmp`). En Render
free, los *warm restarts* son rápidos; los *cold starts* tras inactividad superan
los 30 s.

**Fallback / Cadena de respaldo.** Cuando un sistema intenta primero un método
preferente y, si falla, recurre automáticamente a uno alternativo. En este
proyecto, la API intenta cargar el modelo desde el *Model Registry* y, ante
cualquier error, recurre al *pickle* versionado en el repositorio, registrando el
motivo del fallo en el endpoint `/model-info`.
