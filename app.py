import json
import re
import pandas as pd
import streamlit as st
import fsspec

st.set_page_config(page_title="Análisis de Ventas (GCS)", layout="wide")
st.title("Dashboard de Ventas – XGBoost (lectura desde GCS)")

# ========= CREDENCIALES =========
def _normalize_bucket(b: str) -> str:
    b = (b or "").strip()
    if not b:
        return ""
    if not b.startswith("gs://"):
        b = "gs://" + b
    return b.rstrip("/")

def load_bucket_and_token_from_secrets():
    # Modo A: BUCKET + GCP_SA_JSON
    bucket_a = _normalize_bucket(st.secrets.get("BUCKET", ""))
    sa_json_raw = st.secrets.get("GCP_SA_JSON", "")
    if bucket_a and sa_json_raw:
        try:
            sa_info = json.loads(sa_json_raw)
        except Exception as e:
            st.error(f"GCP_SA_JSON inválido: {e}")
            st.stop()
        return bucket_a, sa_info

    # Modo B: sección [gcp]
    if "gcp" in st.secrets:
        gcp = st.secrets["gcp"]
        need = [
            "project_id","bucket","private_key_id","private_key","client_email",
            "client_id","auth_uri","token_uri","auth_provider_x509_cert_url",
            "client_x509_cert_url"
        ]
        missing = [k for k in need if not gcp.get(k)]
        if missing:
            st.error(f"Faltan claves en [gcp]: {missing}")
            st.stop()
        bucket_b = _normalize_bucket(gcp["bucket"])
        sa_info = {
            "type": "service_account",
            "project_id": gcp["project_id"],
            "private_key_id": gcp["private_key_id"],
            "private_key": gcp["private_key"],
            "client_email": gcp["client_email"],
            "client_id": gcp["client_id"],
            "auth_uri": gcp["auth_uri"],
            "token_uri": gcp["token_uri"],
            "auth_provider_x509_cert_url": gcp["auth_provider_x509_cert_url"],
            "client_x509_cert_url": gcp["client_x509_cert_url"],
            "universe_domain": gcp.get("universe_domain", "googleapis.com"),
        }
        return bucket_b, sa_info

    st.warning("Configura Secrets: (BUCKET+GCP_SA_JSON) o sección [gcp].")
    st.stop()

BUCKET, SA_INFO = load_bucket_and_token_from_secrets()
fs = fsspec.filesystem("gcs", token=SA_INFO)

# ========= HELPERS =========
def _fnmatch(paths, pattern):
    regex = "^" + re.escape(pattern).replace("\\*", ".*").replace("\\?", ".") + "$"
    r = re.compile(regex)
    return [p for p in paths if r.search(p.split("/")[-1])]

def latest_path(prefix: str, pattern: str):
    try:
        paths = fs.glob(f"{prefix}{pattern}")
        if paths:
            return sorted(paths)[-1]
    except Exception:
        pass
    try:
        base = prefix.rstrip("/")
        listed = fs.ls(base)
        candidates = _fnmatch(listed, pattern)
        return sorted(candidates)[-1] if candidates else None
    except Exception:
        return None

def read_json(path: str):
    with fs.open(path, "r") as f:
        return json.load(f)

def read_csv(path: str, parse_dates=None):
    with fs.open(path, "rb") as f:
        return pd.read_csv(f, parse_dates=parse_dates)

# ========= DIAGNÓSTICO =========
with st.expander("Diagnóstico de conexión (click para ver)"):
    st.write("Bucket:", BUCKET)
    try:
        sample = fs.glob(f"{BUCKET}/**")[:10]
        st.write("Primeros objetos:", sample if sample else "(vacío o sin coincidencias)")
    except Exception as e:
        st.error(f"No pude listar el bucket. Detalle: {e}")
        st.stop()

# ========= RESUMEN =========
st.divider()
st.subheader("Resumen del modelo")

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

# ========= DIAGNÓSTICOS =========
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

# ========= PRONÓSTICO =========
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
