# 📓 Notebooks

Los cuadernos del proyecto cuentan una **historia en dos niveles**:

```
playground/  (aprender)   →   src/  (generalizar)   →   API + UI  (mostrar)
```

1. **`playground/` — aprender practicando.** Cuadernos **autónomos** (no importan
   `src/`) que replican el estilo de las clases (`recursos/`) sobre el problema
   real. Es donde experimentamos. Ver [`playground/README.md`](playground/README.md).
2. **`src/ml_hotel_cancellations` — generalizar.** Lo que funcionó en el
   *playground* lo ordenamos y lo convertimos en un **paquete instalable**: la
   *fuente única de verdad* (carga, preprocesado, entrenamiento, inferencia…).
3. **La API + la interfaz — mostrar.** El resultado consolidado ya no se enseña en
   cuadernos finales, sino que se **sirve y se demuestra en vivo**: una API REST
   (FastAPI) que sirve el mejor modelo y una interfaz Streamlit que reúne
   resultados, gráficos, exploración, interpretabilidad (SHAP) y un formulario de
   **predicción que consume la API**.

Así, lo que ves en la web es **exactamente** lo que se entrenó: mismos datos,
mismo preprocesado, mismo modelo. No hay *drift*.

## Cómo verlo en marcha

Desde la raíz del repo, con el entorno instalado (`make setup` o `pip install -e .`):

```bash
uvicorn ml_hotel_cancellations.api.main:app --reload      # 1) la API en :8000 (/docs)
streamlit run src/ml_hotel_cancellations/ui/app.py        # 2) la interfaz
```

O todo a la vez con `make run`. Los gráficos y tablas que muestra la interfaz se
regeneran desde la CLI: `make train` (+ `make viz2d`, `make explain`).

## Estilo (nivel playground)

- **matplotlib/seaborn** (no Plotly): gráficos estáticos que se ven bien en GitHub.
- Cuadernos **autónomos**: no importan `src`, replican las herramientas de clase
  (`pd.get_dummies`, `GridSearchCV`…) para *aprender* las reglas que luego `src`
  generaliza.
- Prosa y comentarios en **español**; identificadores en **inglés**. Términos
  técnicos explicados; glosario en [`../docs/glosario.md`](../docs/glosario.md).

## Cómo ejecutar el playground

Requieren el entorno de desarrollo y el paquete instalado en editable:

```bash
make setup-dev                 # crea .venv e instala [train,dev] (incluye Jupyter)
.venv/bin/jupyter lab          # abrir y ejecutar
```

Detalle de cada cuaderno en [`playground/README.md`](playground/README.md).
