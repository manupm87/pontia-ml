# Interpretabilidad del modelo (bonus)

> ¿No conoces algún término técnico (ROC-AUC, log-odds, one-hot, Pipeline...)? Todos
> están explicados en lenguaje sencillo en [`glosario.md`](glosario.md).

## 1. ¿Por qué interpretar el modelo?

El mejor modelo del proyecto es **XGBoost**, con un **ROC-AUC de 0.9564** en test.
Acierta mucho, pero por sí solo es una **caja negra**: nos da una probabilidad de
cancelación, pero no *por qué*. En un problema real esto no basta. Necesitamos
interpretar el modelo para:

- **Confiar** en él: comprobar que decide por razones sensatas y no por
  casualidades de los datos.
- **Detectar sesgos** o fugas de información (*data leakage*).
- **Explicar** las decisiones al negocio ("esta reserva es de riesgo porque...").

Para ello usamos dos técnicas complementarias, implementadas en el módulo
reutilizable [`src/ml_hotel_cancellations/utils/interpretability.py`](../src/ml_hotel_cancellations/utils/interpretability.py)
(con el *console script* `explain`) y mostradas en la interfaz Streamlit (sección de
interpretabilidad), que renderiza los gráficos SHAP generados por ese módulo.

> Trazabilidad: la exploración inicial de SHAP vive en el *playground*
> [`notebooks/playground/07_interpretabilidad.ipynb`](../notebooks/playground/07_interpretabilidad.ipynb)
> (estilo recursos, autónomo); de ahí se generalizó el módulo reutilizable
> `utils/interpretability.py`, que es lo que consumen la CLI (`explain`) y la UI.

## 2. SHAP: repartir la predicción de forma justa

**SHAP** (*SHapley Additive exPlanations*) se basa en los **valores de Shapley**,
un concepto de la **teoría de juegos cooperativos**. La analogía:

- Las **variables** de una reserva (`deposit_type`, `previous_cancellations`,
  `country`...) son los *jugadores* de un equipo.
- La **predicción** es el *premio* que el equipo consigue.
- El valor de Shapley reparte ese premio entre los jugadores según cuánto aporta
  cada uno, de forma matemáticamente justa.

Aplicado a nuestro modelo, SHAP descompone **cada predicción** en una **suma de
contribuciones**, una por variable:

- Valor SHAP **positivo** → empuja la predicción hacia *"cancela"*.
- Valor SHAP **negativo** → empuja hacia *"no cancela"*.
- El **tamaño** indica la fuerza del empujón.

Como el modelo es un árbol potenciado (XGBoost), usamos `shap.TreeExplainer`, que
calcula los valores de Shapley de forma **exacta y rápida**. Internamente el
modelo es un `Pipeline`: `FeatureBuilder` (crea variables derivadas como
`has_company`, `has_agent` y `noches` a partir de la reserva en crudo) →
`RareCategoryGrouper` (agrupa categorías poco frecuentes) →
`ColumnTransformer` con *one-hot* → `XGBoost`. SHAP necesita la matriz **ya
preprocesada** (categóricas en *one-hot*: **155 columnas** tras el preprocesado),
así que el módulo aplica primero el `preprocessor` y recupera los nombres de
columna con `get_feature_names_out()`. Por eficiencia, se calcula sobre una
**submuestra de 2000 reservas** (da una imagen global estable sin recorrer todo el
conjunto de test).

### 2.1. Importancia global (todo el conjunto)

Dos gráficos resumen qué variables manejan al modelo *en su conjunto*:

- **Barras** (`shap_summary_bar.png`): la **media del valor absoluto de SHAP** por
  variable. Es el resumen más directo de "qué pesa más", sin distinguir
  dirección.
- **Beeswarm / enjambre** (`shap_summary_beeswarm.png`): cada punto es una reserva;
  el **eje X** es su valor SHAP (derecha = hacia *cancela*, izquierda = hacia *no
  cancela*) y el **color**, el valor de la variable (rojo = alto, azul = bajo).
  Añade la **dirección** del efecto que el gráfico de barras no muestra.

### 2.2. Explicación local (una reserva)

El gráfico de **cascada** (*waterfall*, `shap_waterfall_ejemplo1/2.png`) explica
una predicción individual. Parte del **valor base** (la predicción media del
modelo, en *log-odds*) y va sumando/restando la contribución de cada variable
(rojo = hacia *cancela*, azul = hacia *no cancela*) hasta llegar a la predicción
final de esa reserva. Responde a "¿por qué *esta* reserva concreta?".

## 3. Importancia por permutación (contraste agnóstico al modelo)

SHAP, tal como lo usamos aquí, es específico de árboles. Como **contraste
independiente** añadimos la **importancia por permutación** (de scikit-learn,
`permutation_importance.png`): baraja al azar los valores de una variable y mide
cuánto cae el **ROC-AUC**. Si barajar una variable hunde el rendimiento, el
modelo dependía mucho de ella.

Sus ventajas como complemento:

- Es **agnóstica al modelo**: funciona con cualquier estimador (regresión
  logística, red neuronal...), no solo con árboles.
- Atribuye la importancia a las **variables originales** (`deposit_type`,
  `previous_cancellations`...), no a las 155 columnas one-hot expandidas.

## 4. Hallazgos

Las técnicas coinciden en el ranking, lo que refuerza su fiabilidad. Por importancia
de XGBoost (*gain*), las columnas más influyentes son, en orden:

| Variable | Efecto sobre la predicción | Coincide con el EDA |
|---|---|---|
| `deposit_type_Non Refund` | Sube **muchísimo** el riesgo: domina con diferencia | Sí: tasa de cancelación ~99 % |
| `previous_cancellations` | Historial de cancelaciones → más riesgo | Sí: el segundo factor más fuerte |
| `country_PRT` (Portugal) | Discrimina fuerte; PRT cancela más | Sí: mercado dominante del dataset |
| `market_segment_Groups` | El segmento *Groups* sube el riesgo | Coherente con el EDA |
| `has_company` | No tener empresa asociada → **más** riesgo | Sí: hipótesis de "ausencia informativa" |
| `total_of_special_requests` | Más peticiones → **menos** cancelación | Sí: señal de compromiso |

`lead_time` (la antelación) sigue presente y aporta señal, pero **ya no encabeza** el
ranking: el dominio claro es de `deposit_type_Non Refund`, seguido de
`previous_cancellations`. La aparición alta de `has_company` (una variable
**derivada nueva**, creada por `FeatureBuilder`) es especialmente valiosa: confirma
la hipótesis del EDA de la **"ausencia informativa"** — que *no* tener una empresa
asociada a la reserva eleva el riesgo de cancelación. Lo mismo aplica a su gemela
`has_agent`.

> Nota: `required_car_parking_spaces` **ya no forma parte del modelo**. El EDA la
> identificó como una **fuga de información** (*data leakage*) de *check-in* y se
> retiró del pipeline, por lo que no aparece en estos rankings.

**Conclusión clave**: el modelo ha aprendido **exactamente los patrones que el
EDA** ([`playground/01_eda_exploracion.ipynb`](../notebooks/playground/01_eda_exploracion.ipynb)) había señalado como
informativos —`deposit_type`, `previous_cancellations` y las variables de ausencia
informativa `has_company`/`has_agent`—, lo que corrobora las decisiones del análisis
exploratorio. Esto da confianza en que generaliza por las razones correctas.

A nivel **local**, los gráficos *waterfall* de `outputs/` ilustran ambos extremos: una
reserva con depósito *Non Refund*, mucha antelación y desde Portugal se predice como
cancelación casi segura; otra con depósito *No Deposit*, peticiones especiales y con
empresa asociada se predice como no cancelación.

## 5. Cómo reproducirlo

Desde la raíz del repo y con el entorno virtual activado:

```bash
# Regenera todos los gráficos en outputs/ (beeswarm, bar, waterfalls, permutación)
python -m ml_hotel_cancellations.utils.interpretability

# Opciones: tamaño de muestra para SHAP y omitir la permutación (más lenta)
python -m ml_hotel_cancellations.utils.interpretability --sample 2000 --no-permutation
```

La interfaz Streamlit muestra estos mismos gráficos en su sección de interpretabilidad.

## 6. Limitaciones

- SHAP se calcula sobre una **submuestra** (2000 reservas) por eficiencia; la
  imagen global es estable pero no usa el 100 % del test.
- Las variables de **alta cardinalidad** (`country`, `agent`...) se reducen antes
  del *one-hot* con `RareCategoryGrouper`: se conservan solo las categorías con
  soporte suficiente (`RARE_MIN_N`) y señal de cancelación extrema, y el resto se
  agrupa en una etiqueta común `"Otros"`. Por eso su importancia se reparte entre
  las categorías más frecuentes (p. ej. `country_PRT`) y ese grupo `"Otros"`; tras
  el preprocesado quedan **155 columnas** one-hot.
- Los valores SHAP del modelo de árboles se expresan en **log-odds**, no en
  probabilidad directa: indican la **dirección y fuerza** relativa, no un cambio
  porcentual exacto.
- `TreeExplainer` es específico de modelos de árboles; para modelos no basados en
  árboles habría que usar otro explicador de SHAP (p. ej. `KernelExplainer`) o
  apoyarse en la importancia por permutación, que sí es agnóstica.
