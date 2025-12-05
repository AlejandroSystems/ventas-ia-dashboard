import json
import re
import pandas as pd
import streamlit as st
import fsspec

st.set_page_config(page_title="Análisis de Ventas (GCS)", layout="wide")
st.title("Dashboard de Ventas – XGBoost (lectura desde GCS) - Grupo 01 / Univesidad Autónoma del Perú")

# ==========================================
# 1. GESTIÓN DE CREDENCIALES Y CONEXIÓN
# ==========================================
def _normalize_bucket(b: str) -> str:
    b = (b or "").strip()
    if not b:
        return ""
    if not b.startswith("gs://"):
        b = "gs://" + b
    return b.rstrip("/")

def load_bucket_and_token_from_secrets():
    # -------- Modo A: BUCKET + GCP_SA_JSON --------
    bucket_a = _normalize_bucket(st.secrets.get("BUCKET", ""))
    sa_json_raw = st.secrets.get("GCP_SA_JSON", "")

    if bucket_a and sa_json_raw:
        try:
            sa_info = json.loads(sa_json_raw)
        except Exception as e:
            st.error(f"GCP_SA_JSON inválido: {e}")
            st.stop()
        return bucket_a, sa_info

    # -------- Modo B: sección [gcp] en TOML --------
    if "gcp" in st.secrets:
        gcp = st.secrets["gcp"]
        need = [
            "project_id", "bucket", "private_key_id", "private_key",
            "client_email", "client_id", "auth_uri", "token_uri",
            "auth_provider_x509_cert_url", "client_x509_cert_url"
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

    # Si llegó aquí, no hay nada configurado
    st.warning("Configura Secrets: (BUCKET + GCP_SA_JSON) o la sección [gcp].")
    st.stop()

# Cargamos bucket y credenciales
BUCKET, SA_INFO = load_bucket_and_token_from_secrets()

# Filesystem GCS (vía fsspec + gcsfs)
fs = fsspec.filesystem("gcs", token=SA_INFO)

# ==========================================
# 2. FUNCIONES AUXILIARES (HELPERS)
# ==========================================
def _fnmatch(paths, pattern):
    """Filtro simple tipo fnmatch sobre el nombre de archivo."""
    regex = "^" + re.escape(pattern).replace("\\*", ".*").replace("\\?", ".") + "$"
    r = re.compile(regex)
    return [p for p in paths if r.search(p.split("/")[-1])]

def latest_path(prefix: str, pattern: str):
    """
    Devuelve el path más "reciente" (último alfabéticamente) que coincide con pattern
    bajo prefix. Usa glob y, si falla, ls + regex.
    """
    # Intento 1: glob directo
    try:
        paths = fs.glob(f"{prefix}{pattern}")
        if paths:
            return sorted(paths)[-1]
    except Exception:
        pass

    # Intento 2: ls + regex manual
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

# ==========================================
# 3. DIAGNÓSTICO DE CONEXIÓN
# ==========================================
with st.expander("Diagnóstico de conexión (click para ver)"):
    st.write("Bucket:", BUCKET)
    try:
        sample = fs.glob(f"{BUCKET}/**")[:10]
        st.write("Primeros objetos:", sample if sample else "(vacío o sin coincidencias)")
    except Exception as e:
        st.error(f"No pude listar el bucket. Detalle: {e}")
        st.stop()

# ==========================================
# 4. RESUMEN DEL MODELO + CALIDAD DEL DATASET
# ==========================================
st.divider()
st.subheader("Resumen del modelo")

summary_prefix = f"{BUCKET}/reports/"
# Buscamos summary_*.json (p.ej. summary_20251110.json)
summary = latest_path(summary_prefix, "summary_*.json")

if summary is None:
    st.info("No se encontró summary_*.json en reports/")
else:
    try:
        meta = read_json(summary)

        # ===== MÉTRICAS DEL MODELO =====
        c1, c2, c3 = st.columns(3)
        best = meta.get("best_model_snapshot", {})

        rmse = best.get("RMSE", float("nan"))
        mae  = best.get("MAE", float("nan"))
        r2   = best.get("R2", float("nan"))

        c1.metric("RMSE", f"{rmse:.2f}" if rmse == rmse else "—")
        c2.metric("MAE",  f"{mae:.2f}" if mae == mae else "—")
        c3.metric("R²",   f"{r2:.3f}" if r2 == r2 else "—")

        st.caption(
            f"Corte: {meta.get('cutoff_date','?')} • "
            f"Horizonte: {meta.get('horizon_days','?')} días • "
            f"Archivo: {summary.split('/')[-1]}"
        )

        # ===== CALIDAD DEL DATASET =====
        st.markdown("### Calidad del dataset")

        c4, c5, c6 = st.columns(3)
        c4.metric("Filas totales", f"{meta.get('rows', 0):,}".replace(",", " "))
        c5.metric("Series (store+family)", f"{meta.get('n_series', 0):,}".replace(",", " "))
        c6.metric(
            "% series ≥ 12m",
            f"{meta.get('pct_series_ge_12m', 0):.1f}%"
        )

        c7, c8 = st.columns(2)

        # Niveles de cobertura
        with c7:
            st.markdown("**Niveles de cobertura**")
            levels = meta.get("levels_counts", {})
            if levels:
                levels_df = (
                    pd.DataFrame(
                        [{"nivel": k, "series": v} for k, v in levels.items()]
                    )
                    .sort_values("series", ascending=False)
                )
                st.table(levels_df)
            else:
                st.caption("Sin información de levels_counts.")

        # Violaciones de rango / duplicados
        with c8:
            st.markdown("**Anomalías detectadas**")
            rv = meta.get("range_violations", {})
            data_issues = {
                "Ventas negativas": rv.get("sales_negatives", 0),
                "Onpromo negativo": rv.get("onpromo_negatives", 0),
                "Duplicados": meta.get("n_duplicates", 0),
            }
            issues_df = pd.DataFrame(
                [{"tipo": k, "cantidad": v} for k, v in data_issues.items()]
            )
            st.table(issues_df)

    except Exception as e:
        st.error(f"No pude leer/parsing el summary: {e}")

# ==========================================
# 5. DIAGNÓSTICO POR MES Y POR FAMILIA
# ==========================================
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

# ==========================================
# 6. PRONÓSTICO OPERATIVO
# ==========================================
st.divider()
st.subheader("Pronóstico operativo (si existe)")

# Aquí trabajamos con CSV: forecast_*.csv
forecast_csv = latest_path(f"{BUCKET}/reports/", "forecast_*.csv")

if forecast_csv:
    try:
        fc_df = read_csv(forecast_csv, parse_dates=["date"])
        stores = sorted(fc_df["store_nbr"].unique().tolist())
        fams   = sorted(fc_df["family"].unique().tolist())

        s_sel  = st.selectbox("Tienda", stores, index=0)
        f_sel  = st.selectbox("Familia", fams, index=0)

        sdf = (
            fc_df[(fc_df["store_nbr"] == s_sel) & (fc_df["family"] == f_sel)]
            .sort_values("date")
        )

        if not sdf.empty:
            st.line_chart(sdf.set_index("date")[["y_true", "y_pred"]])
            st.caption(f"Archivo: {forecast_csv.split('/')[-1]}")
        else:
            st.info("No hay datos para esa combinación tienda–familia.")

    except Exception as e:
        st.error(f"No pude leer/mostrar forecast: {e}")
else:
    st.info("Aún no hay forecast_*.csv en reports/")

st.divider()
st.caption("Fuente: GCS (solo lectura). Recarga la página si subes nuevos artefactos.")
