import streamlit as st
import pandas as pd
import json
import fsspec

st.set_page_config(page_title="Análisis de Ventas (GCS)", layout="wide")
st.title("Dashboard de Ventas – XGBoost (lectura desde GCS)")

# -------------------------------
# Carga de credenciales (2 modos)
# -------------------------------
def load_bucket_and_token_from_secrets():
    """
    Soporta:
      A) Secrets simples:
         BUCKET = "gs://mi-bucket"
         GCP_SA_JSON = "{...json service account...}"

      B) Secrets TOML con [gcp]:
         [gcp]
         project_id = "..."
         bucket     = "sales-forecast-alejandro-2025"
         private_key_id  = "..."
         private_key     = "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"
         client_email    = "..."
         client_id       = "..."
         auth_uri        = "..."
         token_uri       = "..."
         auth_provider_x509_cert_url = "..."
         client_x509_cert_url        = "..."
         universe_domain = "googleapis.com"
    """
    # --- Modo A: BUCKET + GCP_SA_JSON
    bucket_a = st.secrets.get("BUCKET", "").strip()
    sa_json_raw = st.secrets.get("GCP_SA_JSON", "")

    if bucket_a and sa_json_raw:
        # Normalizar bucket a 'gs://…'
        if not bucket_a.startswith("gs://"):
            bucket_a = "gs://" + bucket_a
        bucket_a = bucket_a.rstrip("/")
        try:
            sa_info = json.loads(sa_json_raw)
        except Exception as e:
            st.error(f"El Secret GCP_SA_JSON no es un JSON válido: {e}")
            st.stop()
        return bucket_a, sa_info

    # --- Modo B: sección [gcp]
    if "gcp" in st.secrets:
        gcp = st.secrets["gcp"]
        # Construir BUCKET
        bucket_b = f"gs://{gcp['bucket']}".rstrip("/")
        # Construir token dict para gcsfs
        sa_info = {
            "type": "service_account",
            "project_id": gcp["project_id"],
            "private_key_id": gcp["private_key_id"],
            "private_key": gcp["private_key"],  # Debe venir con \n en el texto
            "client_email": gcp["client_email"],
            "client_id": gcp["client_id"],
            "auth_uri": gcp["auth_uri"],
            "token_uri": gcp["token_uri"],
            "auth_provider_x509_cert_url": gcp["auth_provider_x509_cert_url"],
            "client_x509_cert_url": gcp["client_x509_cert_url"],
            "universe_domain": gcp.get("universe_domain", "googleapis.com"),
        }
        return bucket_b, sa_info

    # Si no hay ninguna configuración válida:
    st.warning("Faltan Secrets. Usa (BUCKET + GCP_SA_JSON) o la sección [gcp].")
    st.stop()


# Cargar bucket y token
BUCKET, SA_INFO = load_bucket_and_token_from_secrets()

# Crear filesystem GCS (gcsfs vía fsspec)
fs = fsspec.filesystem("gcs", token=SA_INFO)

# -------------------------------
# Helpers
# -------------------------------
def latest_path(prefix: str, pattern: str):
    """Devuelve el path más reciente (por orden alfabético) que cumpla el patrón."""
    try:
        paths = fs.glob(f"{prefix}{pattern}")
    except Exception as e:
        st.error(f"Error listando '{prefix}{pattern}': {e}")
        return None
    if not paths:
        return None
    return sorted(paths)[-1]

# -------------------------------
# Diagnóstico de conexión
# -------------------------------
with st.expander("Diagnóstico de conexión (click para ver)"):
    st.write("Bucket:", BUCKET)
    try:
        sample = fs.glob(f"{BUCKET}/**")[:10]
        st.write("Primeros objetos:", sample if sample else "(vacío o sin coincidencias)")
    except Exception as e:
        st.error(f"No pude listar objetos del bucket '{BUCKET}'. Revisa permisos/Secrets.\nDetalle: {e}")
        st.stop()

st.divider()
st.subheader("Resumen del modelo")

# Espera un JSON tipo: reports/summary_*.json con:
# {
#   "best_model_snapshot": {"RMSE": ..., "MAE": ..., "R2": ...},
#   "cutoff_date": "YYYY-MM-DD",
#   "horizon_days": 30,
#   ...
# }
summary_prefix = f"{BUCKET}/reports/"
summary = latest_path(summary_prefix, "summary_*.json")
if summary is None:
    st.info("No se encontró summary_*.json en reports/")
else:
    try:
        with fs.open(summary, "r") as f:
            meta = json.load(f)
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{meta['best_model_snapshot']['RMSE']:.2f}")
        c2.metric("MAE",  f"{meta['best_model_snapshot']['MAE']:.2f}")
        c3.metric("R²",   f"{meta['best_model_snapshot']['R2']:.3f}")
        st.caption(
            f"Corte: {meta.get('cutoff_date','?')} • "
            f"Horizonte: {meta.get('horizon_days','?')} días • "
            f"Archivo: {summary.split('/')[-1]}"
        )
    except Exception as e:
        st.error(f"No pude leer/parsing el summary: {e}")

st.divider()
st.subheader("Diagnóstico por mes y por familia")

month_csv  = latest_path(f"{BUCKET}/reports/diagnostics/", "month_report_*.csv")
family_csv = latest_path(f"{BUCKET}/reports/diagnostics/", "family_report_*.csv")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Métricas por mes (último run)**")
    if month_csv:
        try:
            with fs.open(month_csv, "rb") as f:
                month_df = pd.read_csv(f)
            st.dataframe(month_df, use_container_width=True)
        except Exception as e:
            st.error(f"No pude leer month_report: {e}")
    else:
        st.info("No se encontró month_report_*.csv")

with c2:
    st.markdown("**Métricas por familia (último run)**")
    if family_csv:
        try:
            with fs.open(family_csv, "rb") as f:
                family_df = pd.read_csv(f)
            st.dataframe(family_df, use_container_width=True)
        except Exception as e:
            st.error(f"No pude leer family_report: {e}")
    else:
        st.info("No se encontró family_report_*.csv")

st.divider()
st.subheader("Pronóstico operativo (si existe)")

forecast_csv = latest_path(f"{BUCKET}/reports/", "forecast_*.csv")
if forecast_csv:
    try:
        with fs.open(forecast_csv, "rb") as f:
            fc_df = pd.read_csv(f, parse_dates=["date"])
        stores = sorted(fc_df["store_nbr"].unique().tolist())
        fams   = sorted(fc_df["family"].unique().tolist())
        s_sel  = st.selectbox("Tienda", stores, index=0)
        f_sel  = st.selectbox("Familia", fams, index=0)
        sdf = fc_df[(fc_df["store_nbr"] == s_sel) & (fc_df["family"] == f_sel)].sort_values("date")
        st.line_chart(sdf.set_index("date")[["y_true", "y_pred"]])
        st.caption(f"Archivo: {forecast_csv.split('/')[-1]}")
    except Exception as e:
        st.error(f"No pude leer/mostrar forecast: {e}")
else:
    st.info("Aún no hay forecast_*.csv en reports/")

st.divider()
st.caption("Fuente: GCS (solo lectura). Recarga la página si subes nuevos artefactos.")
