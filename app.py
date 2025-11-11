import json
import re
from functools import lru_cache

import pandas as pd
import streamlit as st
import fsspec

st.set_page_config(page_title="Análisis de Ventas (GCS)", layout="wide")
st.title("Dashboard de Ventas – XGBoost (lectura desde GCS)")

# ===============================
# CREDENCIALES (BUCKET + TOKEN)
# ===============================

def _normalize_bucket(b: str) -> str:
    b = (b or "").strip()
    if not b:
        return ""
    if not b.startswith("gs://"):
        b = "gs://" + b
    return b.rstrip("/")

def load_bucket_and_token_from_secrets():
    """
    Soporta dos modos de Secrets:
      A) Claves planas:
         BUCKET = "gs://sales-forecast-alejandro-2025"
         GCP_SA_JSON = "{...json service account...}"

      B) Sección TOML:
         [gcp]
         project_id = "rapid-fulcrum-400722"
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
    bucket_a = _normalize_bucket(st.secrets.get("BUCKET", ""))
    sa_json_raw = st.secrets.get("GCP_SA_JSON", "")

    if bucket_a and sa_json_raw:
        try:
            sa_info = json.loads(sa_json_raw)
        except Exception as e:
            st.error(f"El Secret GCP_SA_JSON no es JSON válido: {e}")
            st.stop()
        return bucket_a, sa_info

    # --- Modo B: sección [gcp]
    if "gcp" in st.secrets:
        gcp = st.secrets["gcp"]
        required = [
            "project_id", "bucket", "private_key_id", "private_key", "client_email",
            "client_id", "auth_uri", "token_uri", "auth_provider_x509_cert_url",
            "client_x509_cert_url"
        ]
        missing = [k for k in required if not gcp.get(k)]
        if missing:
            st.error(f"Faltan claves en [gcp] Secrets: {missing}")
            st.stop()

        bucket_b = _normalize_bucket(gcp.get("bucket"))
        sa_info = {
            "type": "service_account",
            "project_id": gcp.get("project_id"),
            "private_key_id": gcp.get("private_key_id"),
            "private_key": gcp.get("private_key"),
            "client_email": gcp.get("client_email"),
            "client_id": gcp.get("client_id"),
            "auth_uri": gcp.get("auth_uri"),
            "token_uri": gcp.get("token_uri"),
            "auth_provider_x509_cert_url": gcp.get("auth_provider_x509_cert_url"),
            "client_x509_cert_url": gcp.get("client_x509_cert_url"),
            "universe_domain": gcp.get("universe_domain", "googleapis.com"),
        }
        return bucket_b, sa_info

    st.warning("Faltan Secrets. Usa (BUCKET + GCP_SA_JSON) o la sección [gcp].")
    st.stop()

BUCKET, SA_INFO = load_bucket_and_token_from_secrets()

# Filesystem GCS (gcsfs vía fsspec)
fs = fsspec.filesystem("gcs", token=SA_INFO)

# ===============================
# HELPERS (glob robusto + cache)
# ===============================

def _fnmatch(paths, pattern):
    # summary_*.json -> regex
    regex = "^" + re.escape(pattern).replace("\\*", ".*").replace("\\?", ".") + "$"
    r = re.compile(regex)
    return [p for p in paths if r.search(p.split("/")[-1])]

def latest_path(prefix: str, pattern: str):
    """Devuelve el path más reciente (orden alfabético) que cumpla el patrón."""
    # Plan A: glob directo
    try:
        paths = fs.glob(f"{prefix}{pattern}")
        if paths:
            return sorted(paths)[-1]
    except Exception:
        pass
    # Plan B: listar y filtrar
    try:
        base = prefix.rstrip("/")
        listed = fs.ls(base)
        candidates = _fnmatch(listed, pattern)
        return sorted(candidates)[-1] if candidates else None
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=60)
def read_json(path: str):
    with fs.open(path, "r") as f:
        return json.load(f)

@st.cache_data(show_spinner=False, ttl=60)
def read_csv(path: str, parse_dates=None):
    with fs.open(path, "rb") as f:
        return pd.read_csv(f, parse_dates=parse_dates)

# ===============================
# DIAGNÓSTICO DE CONEXIÓN
# ===============================

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

# Espera reports/summary_*.json con:
# { "best_model_snapshot": {"RMSE":..., "MAE":..., "R2":...},
#   "cutoff_date":"YYYY-MM-DD", "horizon_days":30, ... }
summary = latest_path(f"{BUCKET}/reports/", "summary_*.json")
if summary:
    try:
        meta = read_json(summary)
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
else:
    st.info("No se encontró summary_*.json en reports/")

# ===============================
# REPORTES DE DIAGNÓSTICO
# ===============================

st.divider()
st.subheader("Diagnóstico por mes y por familia")

month_csv  = latest_path(f"{BUCKET}/reports/diagnostics/", "month_report_*.csv")
family_csv = latest_path(f"{BUCKET}/reports/diagnostics/", "family_report_*.csv")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Métricas por mes (último run)**")
    if month_csv:
        try:
            month_df = read_csv(month_csv)
            st.dataframe(month_df, use_container_width=True)
        except Exception as e:
            st.error(f"No pude leer month_report: {e}")
    else:
        st.info("No se encontró month_report_*.csv")

with c2:
    st.markdown("**Métricas por familia (último run)**")
    if family_csv:
        try:
            family_df = read_csv(family_csv)
            st.dataframe(family_df, use_container_width=True)
        except Exception as e:
            st.error(f"No pude leer family_report: {e}")
    else:
        st.info("No se encontró family_report_*.csv")

# ===============================
# PRONÓSTICO OPERATIVO
# ===============================

st.divider()
st.subheader("Pronóstico operativo (si existe)")

forecast_csv = latest_path(f"{BUCKET}/reports/", "forecast_*.csv")
if forecast_csv:
    try:
        fc_df = read_csv(forecast_csv, parse_dates=["date"])
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
