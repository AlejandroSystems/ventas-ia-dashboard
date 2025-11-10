import streamlit as st
import pandas as pd
import json
import glob
import fsspec

st.set_page_config(page_title="Análisis de Ventas (GCS)", layout="wide")
st.title("Dashboard de Ventas – XGBoost (lectura desde GCS)")

# --- Config mínima (usa Secrets en la nube) ---
BUCKET = st.secrets.get("BUCKET", "")
SA_JSON = st.secrets.get("GCP_SA_JSON", "")

if not BUCKET or not SA_JSON:
    st.warning("Faltan Secrets: BUCKET y/o GCP_SA_JSON")
    st.stop()

# Proveedor de archivos con credenciales (gcsfs)
fs = fsspec.filesystem("gcs", token=json.loads(SA_JSON))

# Helpers
def latest_path(prefix: str, pattern: str):
    paths = fs.glob(f"{prefix}{pattern}")
    if not paths:
        return None
    return sorted(paths)[-1]  # el más reciente por nombre

st.subheader("Resumen del modelo")
summary_prefix = f"{BUCKET}/reports/"
summary = latest_path(summary_prefix, "summary_*.json")
if summary is None:
    st.info("No se encontró summary_*.json en reports/")
else:
    with fs.open(summary, "r") as f:
        meta = json.load(f)
    cols = st.columns(3)
    cols[0].metric("RMSE", f"{meta['best_model_snapshot']['RMSE']:.2f}")
    cols[1].metric("MAE", f"{meta['best_model_snapshot']['MAE']:.2f}")
    cols[2].metric("R²", f"{meta['best_model_snapshot']['R2']:.3f}")
    st.caption(f"Corte: {meta.get('cutoff_date','?')} • Horizonte: {meta.get('horizon_days','?')} días • Archivo: {summary.split('/')[-1]}")

st.divider()
st.subheader("Diagnóstico por mes y por familia")

# Cargar reportes de diagnóstico
month_csv = latest_path(f"{BUCKET}/reports/diagnostics/", "month_report_*.csv")
family_csv = latest_path(f"{BUCKET}/reports/diagnostics/", "family_report_*.csv")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Métricas por mes (último run)**")
    if month_csv:
        with fs.open(month_csv, "rb") as f:
            month_df = pd.read_csv(f)
        st.dataframe(month_df, use_container_width=True)
    else:
        st.info("No se encontró month_report_*.csv")

with c2:
    st.markdown("**Métricas por familia (último run)**")
    if family_csv:
        with fs.open(family_csv, "rb") as f:
            family_df = pd.read_csv(f)
        st.dataframe(family_df, use_container_width=True)
    else:
        st.info("No se encontró family_report_*.csv")

st.divider()
st.subheader("Pronóstico operativo (si existe)")

forecast_csv = latest_path(f"{BUCKET}/reports/", "forecast_*.csv")
if forecast_csv:
    with fs.open(forecast_csv, "rb") as f:
        fc_df = pd.read_csv(f, parse_dates=["date"], infer_datetime_format=True)
    # Filtros
    stores = sorted(fc_df["store_nbr"].unique().tolist())
    fams   = sorted(fc_df["family"].unique().tolist())
    s_sel  = st.selectbox("Tienda", stores, index=0)
    f_sel  = st.selectbox("Familia", fams, index=0)
    sdf = fc_df[(fc_df["store_nbr"]==s_sel)&(fc_df["family"]==f_sel)].sort_values("date")
    st.line_chart(sdf.set_index("date")[["y_true","y_pred"]])
    st.caption(f"Archivo: {forecast_csv.split('/')[-1]}")
else:
    st.info("Aún no hay forecast_*.csv en reports/")

st.divider()
st.caption("Fuente: GCS (solo lectura). Recarga la página si subes nuevos artefactos.")
