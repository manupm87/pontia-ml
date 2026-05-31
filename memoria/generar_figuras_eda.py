"""Genera las figuras del EDA que justifican las decisiones de diseño de la memoria.

Carga el dataset CRUDO (igual que el playground `01_eda_exploracion.ipynb`, antes de
limpiar) para poder mostrar el estado real de los datos en el momento en que se tomó
cada decisión: desbalance de la clase, huecos, ausencia informativa, señal de las
variables numéricas/categóricas, la fuga de `required_car_parking_spaces` y la alta
cardinalidad que motiva la reducción previa al one-hot.

Uso:  .venv/bin/python memoria/generar_figuras_eda.py
Salida: memoria/figuras/eda_*.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RUTA_CSV = RAIZ / "data" / "raw" / "dataset_practica_final.csv"
DIR_FIG = Path(__file__).resolve().parent / "figuras"
DIR_FIG.mkdir(parents=True, exist_ok=True)

# Estilo coherente para toda la memoria.
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})
AZUL = "#3b6ea5"      # no cancelada
ROJO = "#c0504d"      # cancelada
GRIS = "#8c8c8c"


def guardar(fig: plt.Figure, nombre: str) -> None:
    ruta = DIR_FIG / nombre
    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {ruta.relative_to(RAIZ)}")


def main() -> None:
    df = pd.read_csv(RUTA_CSV, na_values=["NULL", "NA", "NaN", ""])
    df["has_company"] = df["company"].notna()
    df["has_agent"] = df["agent"].notna()
    print(f"Cargadas {len(df):,} reservas, {df.shape[1]} columnas")

    # ── 1. Desbalance de la variable objetivo ──────────────────────────────
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    conteo = df["is_canceled"].value_counts().sort_index()
    pct = 100 * conteo / conteo.sum()
    barras = ax.bar(["No cancelada (0)", "Cancelada (1)"], conteo.values,
                    color=[AZUL, ROJO])
    for b, p, n in zip(barras, pct.values, conteo.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{p:.1f}%\n({n:,})", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Nº de reservas")
    ax.set_title("Distribución de la clase objetivo `is_canceled`")
    ax.set_ylim(0, conteo.max() * 1.18)
    guardar(fig, "eda_desbalance_clase.png")

    # ── 2. Valores ausentes por columna ────────────────────────────────────
    nulos = (df.isna().mean() * 100).sort_values(ascending=False)
    nulos = nulos[nulos > 0]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    barras = ax.barh(nulos.index[::-1], nulos.values[::-1], color=GRIS)
    for b, v in zip(barras, nulos.values[::-1]):
        ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("% de valores ausentes")
    ax.set_title("Huecos (NaN) por columna")
    ax.set_xlim(0, 105)
    guardar(fig, "eda_valores_ausentes.png")

    # ── 3. Ausencia informativa: has_company / has_agent ───────────────────
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.3), sharey=True)
    for ax, col, etq in [(axes[0], "has_company", "company"),
                         (axes[1], "has_agent", "agent")]:
        tasa = df.groupby(col)["is_canceled"].mean() * 100
        ax.bar([f"Sin {etq}", f"Con {etq}"],
               [tasa.get(False, 0), tasa.get(True, 0)], color=[ROJO, AZUL])
        for x, v in enumerate([tasa.get(False, 0), tasa.get(True, 0)]):
            ax.text(x, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=10)
        ax.set_title(f"Tasa según `{col}`")
    axes[0].set_ylabel("% cancelación")
    fig.suptitle("Ausencia informativa: el hueco es señal, no ruido", y=1.02)
    guardar(fig, "eda_ausencia_informativa.png")

    # ── 4. lead_time: la numérica más predictiva ───────────────────────────
    bins = [-1, 7, 30, 90, 180, 365, 10000]
    etqs = ["0-7", "8-30", "31-90", "91-180", "181-365", "365+"]
    df["_lt"] = pd.cut(df["lead_time"], bins=bins, labels=etqs)
    tasa = df.groupby("_lt", observed=True)["is_canceled"].mean() * 100
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    ax.plot(tasa.index.astype(str), tasa.values, "-o", color=ROJO, lw=2)
    for x, v in enumerate(tasa.values):
        ax.text(x, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
    ax.set_xlabel("Antelación de la reserva (días)")
    ax.set_ylabel("% cancelación")
    ax.set_title("`lead_time`: a más antelación, más cancelación")
    ax.set_ylim(0, tasa.max() * 1.18)
    guardar(fig, "eda_lead_time.png")

    # ── 5. deposit_type: la categórica más fuerte ──────────────────────────
    tasa = (df.groupby("deposit_type")["is_canceled"].mean() * 100).sort_values()
    n = df["deposit_type"].value_counts()
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    colores = [ROJO if v > 50 else AZUL for v in tasa.values]
    barras = ax.bar(tasa.index, tasa.values, color=colores)
    for b, v, k in zip(barras, tasa.values, tasa.index):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}%\n(n={n[k]:,})",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("% cancelación")
    ax.set_title("`deposit_type`: 'Non Refund' cancela casi siempre")
    ax.set_ylim(0, 112)
    guardar(fig, "eda_deposit_type.png")

    # ── 6. required_car_parking_spaces: fuga de información ─────────────────
    df["_park"] = np.where(df["required_car_parking_spaces"] > 0, "≥1 plaza", "0 plazas")
    glob = df.groupby("_park")["is_canceled"].mean() * 100
    nd = df[df["deposit_type"] == "No Deposit"]
    sub = nd.groupby("_park")["is_canceled"].mean() * 100
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    x = np.arange(2)
    w = 0.38
    b1 = ax.bar(x - w / 2, [glob.get("0 plazas", 0), glob.get("≥1 plaza", 0)], w,
                label="Todas las reservas", color=GRIS)
    b2 = ax.bar(x + w / 2, [sub.get("0 plazas", 0), sub.get("≥1 plaza", 0)], w,
                label="Solo 'No Deposit' (cancelables)", color=AZUL)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4,
                    f"{b.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["0 plazas", "≥1 plaza"])
    ax.set_ylabel("% cancelación")
    ax.set_title("`required_car_parking_spaces`: 0 % es fuga, no señal")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, max(glob.max(), 5) * 1.25)
    guardar(fig, "eda_fuga_parking.png")

    # ── 7. Alta cardinalidad → reducción antes del one-hot ──────────────────
    cols = ["country", "agent", "company"]
    n_unique = [df[c].nunique(dropna=True) for c in cols]
    # nº aproximado de columnas tras la reducción supervisada (de la memoria/§13)
    n_reducidas = [16, 57, 8]
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    x = np.arange(len(cols))
    w = 0.38
    b1 = ax.bar(x - w / 2, n_unique, w, label="Categorías originales", color=ROJO)
    b2 = ax.bar(x + w / 2, n_reducidas, w, label="Columnas tras reducir", color=AZUL)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{int(b.get_height())}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(cols)
    ax.set_ylabel("Nº de categorías / columnas")
    ax.set_title("Alta cardinalidad: reducir antes del OneHotEncoder")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(n_unique) * 1.15)
    guardar(fig, "eda_cardinalidad.png")

    print("Figuras del EDA generadas correctamente.")


if __name__ == "__main__":
    main()
