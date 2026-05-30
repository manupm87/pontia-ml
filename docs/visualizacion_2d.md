# Visualización 2D de las regiones de decisión (proyección PLS)

> ¿No conoces algún término técnico (PLS, PCA, one-hot, Pipeline...)? Muchos están
> explicados en [`glosario.md`](glosario.md).

Este documento explica la visualización 2D del proyecto: cómo pasamos de modelos
entrenados en **155 variables** a un mapa plano donde se *ven* las **regiones de
decisión** de los cinco modelos. El código vive en
[`src/ml_hotel_cancellations/utils/visualization_2d.py`](../src/ml_hotel_cancellations/utils/visualization_2d.py)
(*console script* `viz2d`; `python -m ml_hotel_cancellations.utils.visualization_2d`).

> Trazabilidad de los notebooks (dos niveles): esta visualización se prototipó en el
> *playground* [`notebooks/playground/05_comparativa_y_visualizacion.ipynb`](../notebooks/playground/05_comparativa_y_visualizacion.ipynb)
> (estilo recursos, autónomo) y de ahí se **generalizó** al módulo reutilizable
> `utils/visualization_2d.py`. Es el arco del proyecto:
> **playground (aprender) → `src` (generalizar) → notebooks (mostrar)**.

## 1. El problema: no se puede dibujar en 155 dimensiones

Tras el preprocesado, cada reserva es un vector de **155 variables** (numéricas
escaladas + categóricas en *one-hot*). No podemos dibujar 155 ejes, así que para
*ver* cómo separan los modelos necesitamos **proyectar a 2 dimensiones**.

El enfoque ingenuo —elegir 2 variables— pierde casi toda la información: la señal
que separa cancela/no-cancela está repartida entre muchas columnas (sobre todo
categóricas como `deposit_type` o `country`). La alternativa correcta es una
**proyección** que combine las 155 variables en 2 ejes nuevos conservando lo que
importa.

## 2. PLS supervisado (no PCA)

¿Por qué **PLS** (*Partial Least Squares*) y no PCA?

- **PCA** busca las direcciones de **máxima varianza** de las `X` — **ignora el
  target**. A menudo la varianza grande no es la que separa las clases, y cancela /
  no-cancela salen mezcladas (en este dataset, una proyección PCA 2D apenas separa).
- **PLS** busca las direcciones de `X` **más correlacionadas con `y`**. Es una
  proyección **supervisada**: usa la etiqueta, así que tiende a **separar las
  clases** en el plano. El eje 1 acaba siendo, en la práctica, un *índice de riesgo*
  de cancelación.

### Cómo se construye la proyección (sin fuga)

La matriz de 155 variables que alimenta PLS sale del **mismo preprocesado** del
pipeline de producción, ajustado **con el target** (porque incluye la reducción de
cardinalidad **supervisada** `RareCategoryGrouper`, que necesita `y`):

```python
from ml_hotel_cancellations.ml.preprocessing import build_transform_pipeline
from sklearn.cross_decomposition import PLSRegression

# fit-on-train CON y: la reducción de cardinalidad es supervisada
preprocessor = build_transform_pipeline().fit(X_train, y_train)
Z_train = preprocessor.transform(X_train)   # (n, 155)
pls = PLSRegression(n_components=2).fit(Z_train, y_train)   # 155 -> 2
```

> Antes el módulo usaba `build_preprocessor()` (solo el `ColumnTransformer`). Ahora
> usa `build_transform_pipeline()` (los 3 pasos: `FeatureBuilder` +
> `RareCategoryGrouper` + `ColumnTransformer`) y lo ajusta con `(X_train, y_train)`,
> para que la proyección parta exactamente de las mismas 155 features que ve el
> modelo. El signo del eje 1 se orienta hacia "más riesgo" para que la lectura sea
> siempre la misma (rojo a la derecha).

## 3. Las cinco regiones de decisión

Sobre el plano de 2 componentes PLS se **reentrena** cada uno de los cinco modelos
(Regresión Logística, Árbol de Decisión, Random Forest, XGBoost y una red neuronal
`MLPClassifier` de scikit-learn —no Keras— porque aquí solo aprende de 2 variables).
Para cada modelo se pinta su **región de decisión**: el color de fondo es la
probabilidad de cancelación que asigna a cada punto del plano (rojo = "aquí se
cancela", azul = "aquí no"), con la frontera 0.5 marcada en negro, y encima una
muestra de reservas reales de test.

Así se ve la *personalidad* de cada modelo: la **regresión logística** traza una
frontera casi recta (es lineal); el **árbol** corta en bloques rectangulares;
**Random Forest** y **XGBoost** dibujan fronteras suaves y curvas que envuelven
mejor la zona de cancelación; la **red neuronal** también curva, de forma más
blanda. Es la misma jerarquía que se ve en las métricas, ahora a la vista.

> ⚠️ El plano 2D es para **entender**, no para producir: cada modelo aquí se
> reentrena con solo 2 componentes y rinde algo peor que con las 155. El modelo de
> producción (XGBoost, **ROC-AUC 0.9564**) usa las 155 features completas.

## 4. Artefactos y uso en la interfaz

`viz2d` precalcula y **persiste** dos artefactos en `outputs/`:

- `decision_regions_pls.png` — la figura con las regiones de decisión de los 5
  modelos (recurso visual para documentación y la UI).
- `decision_regions_pls.pkl` — el preprocesador ajustado, el objeto PLS, el signo de
  orientación, la rejilla de probabilidades precomputada y la submuestra de test.
  **Se versiona en el repo** porque la interfaz Streamlit lo carga en tiempo de
  ejecución para **situar la reserva del usuario** sobre el plano (una estrella sobre
  las regiones), sin tener que reentrenar nada (lo que tardaría ~45 s).

Regenerar los artefactos:

```bash
python -m ml_hotel_cancellations.utils.visualization_2d   # o: make viz2d
```

## 5. Limitaciones

- Es una **proyección**: 2 componentes PLS no capturan toda la información de las
  155 variables, así que las fronteras 2D son una **aproximación** a lo que el modelo
  hace en el espacio completo.
- Las métricas de referencia del proyecto provienen del pipeline sobre las 155
  variables, **no** de estos modelos 2D (que rinden algo menos por diseño).
- PLS usa el target: se ajusta **solo con `train`** y se aplica a `test` con esa
  misma transformación; si se ajustara con todo el dataset, habría fuga.
