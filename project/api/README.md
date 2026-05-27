# API REST con FastAPI (bonus)

Esta API expone el mejor modelo entrenado del proyecto (XGBoost dentro de un
`Pipeline` de scikit-learn) a través de HTTP, para poder predecir si una reserva
de hotel será cancelada. Está pensada para ser consumida por una interfaz
externa (por ejemplo, una app de Streamlit).

## Estructura (modular y didáctica)

```
api/
├── __init__.py      # documentación del paquete
├── schemas.py       # contratos de entrada/salida (Pydantic)
├── service.py       # carga del modelo (1 sola vez) + lógica de predicción
├── main.py          # la app FastAPI con los endpoints
├── README.md        # este fichero
└── tests/
    └── test_api.py  # pruebas con TestClient (pytest)
```

El preprocesado de la entrada se hace **reutilizando** `src.predict` y
`src.data_loader.normalize_categoricals`, de modo que la API "ve" los datos
exactamente igual que el entrenamiento (misma normalización de categóricas,
`agent` como texto, etc.).

## Requisitos previos

- El modelo entrenado debe existir en `models/best_model.pkl`
  (genéralo con `python -m src.train` si no está).
- Dependencias instaladas (ver `requirements.txt`): `fastapi`, `uvicorn[standard]`.

## Cómo arrancar el servidor

Desde la carpeta `project/`:

```bash
uvicorn api.main:app --reload
```

Por defecto sirve en http://127.0.0.1:8000. La documentación interactiva
(Swagger UI) está en:

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc:      http://127.0.0.1:8000/redoc

Para usar otro modelo, define la variable de entorno `PONTIA_MODEL_PATH` con la
ruta al `.pkl` antes de arrancar.

## Ejemplo con `curl`

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hotel": "City Hotel",
    "lead_time": 100,
    "arrival_date_month": "August",
    "arrival_date_week_number": 33,
    "arrival_date_day_of_month": 15,
    "stays_in_weekend_nights": 2,
    "stays_in_week_nights": 5,
    "adults": 2,
    "children": 0,
    "babies": 0,
    "meal": "BB",
    "country": "PRT",
    "market_segment": "Online TA",
    "distribution_channel": "TA/TO",
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
    "reserved_room_type": "A",
    "assigned_room_type": "A",
    "booking_changes": 0,
    "deposit_type": "No Deposit",
    "agent": "9",
    "days_in_waiting_list": 0,
    "customer_type": "Transient",
    "adr": 100.0,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 1
  }'
```

Respuesta:

```json
{"prediction": 1, "label": "Cancelada", "probability": 0.87}
```

## Contrato de la API

| Método | Ruta             | Descripción                                  |
|--------|------------------|----------------------------------------------|
| GET    | `/`              | Mensaje de bienvenida + enlace a `/docs`.    |
| GET    | `/health`        | Sondeo de salud del servicio.                |
| GET    | `/model-info`    | Metadatos del modelo y características.       |
| POST   | `/predict`       | Predicción para una reserva.                 |
| POST   | `/predict/batch` | Predicción para una lista de reservas.       |

### `GET /health`

```json
{"status": "ok", "model_loaded": true}
```

### `GET /model-info`

```json
{
  "model_type": "XGBoost",
  "primary_metric": "roc_auc",
  "roc_auc": 0.9614,
  "n_features": 27,
  "features": {"numeric": ["lead_time", "..."], "categorical": ["hotel", "..."]}
}
```

### `POST /predict`

- **Entrada**: una reserva (`Booking`) con las 27 características crudas.
- **Salida** (`200`):

```json
{"prediction": 0, "label": "No cancelada", "probability": 0.12}
```

`prediction` es `0` (No cancelada) o `1` (Cancelada). `probability` es la
probabilidad de cancelación P(cancela), siempre en el rango `[0, 1]`.

### `POST /predict/batch`

- **Entrada**: `{"bookings": [Booking, ...]}`.
- **Salida** (`200`):

```json
{"predictions": [{"prediction": 0, "label": "No cancelada", "probability": 0.12}]}
```

Las predicciones se devuelven en el **mismo orden** que las reservas de entrada.

### Errores de validación

Si falta un campo obligatorio o el tipo es incorrecto, FastAPI responde con
`422 Unprocessable Entity` y un detalle del error.

## Pruebas

Desde `project/`:

```bash
python -m pytest api/tests -q
```
