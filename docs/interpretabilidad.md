# Interpretabilidad del modelo (bonus)

> ¿No conoces algún término técnico (ROC-AUC, log-odds, one-hot, Pipeline...)? Todos
> están explicados en lenguaje sencillo en [`glosario.md`](glosario.md).

## 1. ¿Por qué interpretar el modelo?

El mejor modelo del proyecto es **XGBoost**, con un **ROC-AUC de 0.9614** en test.
Acierta mucho, pero por sí solo es una **caja negra**: nos da una probabilidad de
cancelación, pero no *por qué*. En un problema real esto no basta. Necesitamos
interpretar el modelo para:

- **Confiar** en él: comprobar que decide por razones sensatas y no por
  casualidades de los datos.
- **Detectar sesgos** o fugas de información (*data leakage*).
- **Explicar** las decisiones al negocio ("esta reserva es de riesgo porque...").

Para ello usamos dos técnicas complementarias, implementadas en el módulo
reutilizable [`src/interpretability.py`](../src/interpretability.py) y mostradas
de forma didáctica en el notebook
[`notebooks/10_interpretabilidad_shap.ipynb`](../notebooks/10_interpretabilidad_shap.ipynb).

## 2. SHAP: repartir la predicción de forma justa

**SHAP** (*SHapley Additive exPlanations*) se basa en los **valores de Shapley**,
un concepto de la **teoría de juegos cooperativos**. La analogía:

- Las **variables** de una reserva (`lead_time`, `deposit_type`, `country`...) son
  los *jugadores* de un equipo.
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
modelo es un `Pipeline(preprocessor, model)`; SHAP necesita la matriz **ya
preprocesada** (numéricas estandarizadas + categóricas en *one-hot*), así que el
módulo aplica primero el `preprocessor` y recupera los nombres de columna con
`get_feature_names_out()`. Por eficiencia, se calcula sobre una **submuestra de
2000 reservas** (da una imagen global estable sin recorrer las 23 842 del test).

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
- Atribuye la importancia a las **variables originales** (`lead_time`,
  `deposit_type`...), no a las columnas one-hot expandidas.

## 4. Hallazgos

Las técnicas coinciden en el ranking, lo que refuerza su fiabilidad. Las variables
más influyentes son:

| Variable | Efecto sobre la predicción | Coincide con el EDA |
|---|---|---|
| `deposit_type = "Non Refund"` | Sube **mucho** el riesgo de cancelación | Sí: tasa de cancelación ~99 % |
| `country` (p. ej. `PRT`, Portugal) | Discrimina fuerte; PRT cancela más | Sí: mercado dominante del dataset |
| `lead_time` (antelación) | Más antelación → más cancelación | Sí: la numérica más relacionada |
| `total_of_special_requests` | Más peticiones → **menos** cancelación | Sí: señal de compromiso |
| `required_car_parking_spaces` | Pedir aparcamiento → menos cancelación | Sí: señal de compromiso |
| `previous_cancellations` | Historial de cancelaciones → más riesgo | Coherente |

**Conclusión clave**: el modelo ha aprendido **exactamente los patrones que el
EDA** ([`01_eda.ipynb`](../notebooks/01_eda.ipynb)) había señalado como
informativos. Esto da confianza en que generaliza por las razones correctas.

A nivel **local**, los dos ejemplos del notebook ilustran ambos extremos: una
reserva con depósito *Non Refund*, 379 días de antelación y desde Portugal se
predice como cancelación casi segura; otra con depósito *No Deposit*, peticiones
especiales y desde otro país se predice como no cancelación.

## 5. Cómo reproducirlo

Desde la raíz del repo y con el entorno virtual activado:

```bash
# Regenera todos los gráficos en outputs/ (beeswarm, bar, waterfalls, permutación)
python -m src.interpretability

# Opciones: tamaño de muestra para SHAP y omitir la permutación (más lenta)
python -m src.interpretability --sample 2000 --no-permutation
```

El notebook didáctico se ejecuta de principio a fin con sus salidas renderizadas.

## 6. Limitaciones

- SHAP se calcula sobre una **submuestra** (2000 reservas) por eficiencia; la
  imagen global es estable pero no usa el 100 % del test.
- Las variables de **alta cardinalidad** (`country` 178 valores, `agent` 334) se
  agrupan con el *one-hot* limitado (máx. 25 categorías), por lo que su
  importancia se reparte entre las categorías más frecuentes y una categoría
  "infrequent".
- Los valores SHAP del modelo de árboles se expresan en **log-odds**, no en
  probabilidad directa: indican la **dirección y fuerza** relativa, no un cambio
  porcentual exacto.
- `TreeExplainer` es específico de modelos de árboles; para modelos no basados en
  árboles habría que usar otro explicador de SHAP (p. ej. `KernelExplainer`) o
  apoyarse en la importancia por permutación, que sí es agnóstica.
