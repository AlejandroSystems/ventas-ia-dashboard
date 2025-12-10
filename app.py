import json
import re
import pandas as pd
import numpy as np
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
summary = latest_path(summary_prefix, "summary_*.json")

if summary is None:
    st.info("No se encontró summary_*.json en reports/")
else:
    try:
        meta = read_json(summary)
    except Exception as e:
        st.error(f"No pude leer/parsing el summary: {e}")
    else:
        # ===== MÉTRICAS DEL MODELO =====
        st.markdown("### Métricas del modelo principal")

        metrics = meta.get("metrics", {}) or {}
        candidates = metrics.get("candidates", {}) or {}

        best_name_cfg = meta.get("best_model_name")
        best_name = None
        best_metrics = None

        if candidates:
            if isinstance(best_name_cfg, str) and best_name_cfg in candidates:
                best_name = best_name_cfg
                best_metrics = candidates[best_name]
            else:
                # Elegir el de menor RMSE si no está definido explícitamente
                best_name, best_metrics = sorted(
                    candidates.items(),
                    key=lambda kv: kv[1].get("RMSE", float("inf"))
                )[0]

        rmse = best_metrics.get("RMSE", float("nan")) if best_metrics else float("nan")
        mae  = best_metrics.get("MAE", float("nan")) if best_metrics else float("nan")
        r2   = best_metrics.get("R2", float("nan")) if best_metrics else float("nan")

        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{rmse:.2f}" if rmse == rmse else "—")
        c2.metric("MAE",  f"{mae:.2f}" if mae == mae else "—")
        c3.metric("R²",   f"{r2:.3f}" if r2 == r2 else "—")

        st.caption(
            f"Modelo elegido: {best_name or 'desconocido'} • "
            f"Corte: {meta.get('cutoff_date','?')} • "
            f"Horizonte: {meta.get('horizon_days','?')} días • "
            f"Archivo: {summary.split('/')[-1]}"
        )

        # ----- Interpretación rápida de las métricas -----
        if rmse == rmse and mae == mae and r2 == r2:  # chequeo simple de NaN
            st.markdown(
                f"""
                **Interpretación rápida de las métricas**

                - **RMSE ≈ {rmse:,.0f}** unidades: error típico que penaliza más los errores grandes.  
                - **MAE ≈ {mae:,.0f}** unidades: desvío promedio entre la venta real y la pronosticada
                  por día y combinación tienda–familia.  
                - **R² = {r2:.3f}**: el modelo explica aproximadamente el **{r2*100:.1f}%** de la variación
                  de las ventas.
                """
            )

            if r2 >= 0.90:
                quality = "Excelente (modelo muy explicativo)."
            elif r2 >= 0.70:
                quality = "Aceptable (útil, pero con margen de mejora)."
            else:
                quality = "Limitado (sirve como referencia, no para decisiones críticas)."

            st.caption(f"Evaluación global del modelo: {quality}")
        else:
            st.caption("Métricas incompletas: no es posible generar una interpretación automática.")

        # ===== CALIDAD DEL DATASET =====
        st.markdown("### Calidad del dataset")

        # Data quality anidada en master_summary["data_quality"]
        dq = meta.get("data_quality") or meta

        rows     = int(dq.get("rows", 0))
        n_series = int(dq.get("n_series", 0))
        pct12m   = float(dq.get("pct_series_ge_12m", 0.0))

        c4, c5, c6 = st.columns(3)
        c4.metric("Registros totales (filas)", f"{rows:,}".replace(",", " "))
        c5.metric("Series tienda–familia", f"{n_series:,}".replace(",", " "))
        c6.metric(
            "Series con ≥ 12 meses de historial",
            f"{pct12m:.1f}%"
        )

        st.markdown(
            f"""
            **¿Qué significan estos indicadores de calidad?**

            - **Registros totales:** representan todas las filas históricas de ventas que alimentan el modelo.  
            - **Series tienda–familia:** cada combinación *tienda + familia de producto* se trata como una serie de tiempo independiente.  
            - **Series con ≥ 12 meses:** el **{pct12m:.1f}%** de las series tienen al menos un año de historial, lo que permite detectar estacionalidad anual con mucha más confianza.
            """
        )

        if pct12m >= 90:
            cov_msg = "Cobertura histórica **excelente**: el dataset es muy sólido para análisis de estacionalidad."
        elif pct12m >= 70:
            cov_msg = "Cobertura histórica **aceptable**: se puede trabajar, pero algunas series tendrán menos contexto temporal."
        else:
            cov_msg = "Cobertura histórica **limitada**: las predicciones en ciertas series deben interpretarse con cautela."

        st.caption(cov_msg)

        c7, c8 = st.columns(2)

        # Niveles de cobertura
        with c7:
            st.markdown("**Historial disponible por serie de ventas**")

            levels = dq.get("levels_counts", {}) or {}
            total_series = int(dq.get("n_series", 0))

            if levels and total_series > 0:
                levels_df = (
                    pd.DataFrame(
                        [{"Categoría de cobertura": k, "N.º de series": v} for k, v in levels.items()]
                    )
                    .sort_values("N.º de series", ascending=False)
                )
                # Agregamos porcentaje
                levels_df["% del total"] = (
                    (levels_df["N.º de series"] / total_series * 100)
                    .round(0)
                    .astype(int)
                )

                st.table(levels_df)

                if len(levels_df) == 1 and "óptimo" in str(levels_df["Categoría de cobertura"].iloc[0]).lower():
                    st.caption(
                        f"Las {total_series:,} series cuentan con entre 24 y 60 meses de historial, "
                        "lo que ofrece una base sólida para detectar patrones de estacionalidad y tendencia."
                    )
                else:
                    st.caption(
                        "Un mayor porcentaje en categorías como **óptimo** o **recomendado** indica "
                        "mejor calidad histórica para entrenar modelos robustos."
                    )
            else:
                st.caption("No se encontró información de cobertura histórica por serie.")

        # Violaciones de rango / duplicados
        with c8:
            st.markdown("**Anomalías detectadas en los datos**")
            rv = dq.get("range_violations", {}) or {}
            data_issues = {
                "Ventas negativas": rv.get("sales_negatives", 0),
                "Onpromotion negativo": rv.get("onpromo_negatives", 0),
                "Duplicados (misma fecha y serie)": dq.get("n_duplicates", 0),
            }
            issues_df = pd.DataFrame(
                [{"tipo": k, "cantidad": v} for k, v in data_issues.items()]
            )
            st.table(issues_df)

# ==========================================
# 5. COMPARACIÓN DE MODELOS (IA)
# ==========================================
st.divider()
st.subheader("Comparación de modelos (IA)")

metrics_path = f"{BUCKET}/reports/cv_results.csv"

try:
    metrics_df = read_csv(metrics_path)
    if metrics_df is None or metrics_df.empty:
        raise ValueError("cv_results.csv vacío")
    # Ordenamos por RMSE (menor es mejor)
    metrics_df_sorted = metrics_df.sort_values("RMSE", ascending=True).reset_index(drop=True)

    st.markdown("**Tabla comparativa de modelos (ordenada por RMSE):**")
    st.dataframe(metrics_df_sorted, use_container_width=True)

    best_row = metrics_df_sorted.iloc[0]
    best_name = best_row["model"]
    best_rmse = best_row["RMSE"]
    best_mae  = best_row["MAE"]
    best_r2   = best_row.get("R2", float("nan"))

    st.markdown(
        f"""
        **Mejor modelo actual:** `{best_name}`  
        - RMSE: **{best_rmse:,.2f}**  
        - MAE: **{best_mae:,.2f}**  
        - R²: **{best_r2:.3f}**
        """
    )
except Exception:
    st.info(
        "Aún no se encontró `cv_results.csv` en la carpeta `reports/`. "
        "Ejecuta el Bloque 5 del notebook para generar las métricas de modelos."
    )

# ==========================================
# 6. MUESTREO Y COBERTURA DEL DATASET
# ==========================================
st.divider()
st.subheader("Muestreo y cobertura del dataset")

month_csv  = latest_path(f"{BUCKET}/reports/diagnostics/", "month_report_*.csv")
family_csv = latest_path(f"{BUCKET}/reports/diagnostics/", "family_report_*.csv")

month_df  = None
family_df = None

# Cargamos, pero sin matar la app si falla algo
if month_csv:
    try:
        month_df = read_csv(month_csv)
    except Exception as e:
        st.error(f"No pude leer month_report: {e}")

if family_csv:
    try:
        # start / end vienen como fechas en el CSV de diagnóstico
        family_df = read_csv(family_csv, parse_dates=["start", "end"])
    except Exception as e:
        st.error(f"No pude leer family_report: {e}")

c1, c2 = st.columns(2)

# ------------------------------
# Columna izquierda: muestreo temporal
# ------------------------------
with c1:
    st.markdown("**Resumen de muestreo temporal**")

    if family_df is not None and not family_df.empty:
        global_start = family_df["start"].min()
        global_end   = family_df["end"].max()

        # meses aproximados de histórico
        n_months = (global_end.year - global_start.year) * 12 + (global_end.month - global_start.month) + 1

        st.metric("Meses con historial de ventas", f"{n_months}")
        st.caption(
            f"El dataset cubre ventas diarias desde **{global_start.date()}** "
            f"hasta **{global_end.date()}**, ofreciendo una muestra continua "
            "para analizar estacionalidad y tendencias."
        )
    else:
        st.info("No se encontró información de cobertura temporal (family_report_*.csv).")

    # Calidad del muestreo (valores nulos)
    if month_df is not None and not month_df.empty and "n_nulls" in month_df.columns:
        total_nulls = int(month_df["n_nulls"].sum())
        st.markdown("**Calidad del muestreo (valores nulos)**")
        st.write(
            f"- Valores nulos en columnas clave (fecha y ventas): **{total_nulls}** "
            "(un valor 0 indica que el historial está completo y sin huecos)."
        )
    else:
        st.info("No se encontró month_report_*.csv para revisar nulos en el historial.")

# ------------------------------
# Columna derecha: cobertura por serie
# ------------------------------
with c2:
    st.markdown("**Cobertura por tienda / familia**")

    if family_df is not None and not family_df.empty:
        min_days = int(family_df["days"].min())
        max_days = int(family_df["days"].max())
        pct_ge_24m = (family_df["days"] >= 730).mean() * 100  # 24 meses ≈ 730 días

        st.markdown(
            "- **¿Qué es una serie?** Cada serie corresponde a la evolución diaria de "
            "ventas de **una tienda** para **una familia de productos**.\n"
            "- **¿Por qué importa?** Cuantas más series y con más años de datos, "
            "mejor puede el modelo aprender patrones distintos entre tiendas y categorías."
        )

        st.metric(
            "Series de venta independientes (tienda × familia)",
            f"{len(family_df):,}".replace(",", " ")
        )
        st.metric(
            "Series con historial largo (≥ 24 meses de datos)",
            f"{pct_ge_24m:.0f}%"
        )
        st.caption(
            "Si este porcentaje está cerca de **100 %**, casi todas las series tienen al "
            "menos dos años de historial, lo que reduce el riesgo de decisiones basadas "
            "en muestras cortas y hace al modelo más estable."
        )

        with st.expander("Ver detalle por tienda y familia"):
            st.dataframe(family_df, use_container_width=True)
    else:
        st.info("No se encontró family_report_*.csv para calcular la cobertura por serie.")

# ==========================================
# 7. PRONÓSTICO OPERATIVO
# ==========================================
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

        sdf = (
            fc_df[(fc_df["store_nbr"] == s_sel) & (fc_df["family"] == f_sel)]
            .sort_values("date")
        )

        if not sdf.empty:
            # Rango dinámico según los datos disponibles
            min_date = sdf["date"].min().date()
            max_date = sdf["date"].max().date()

            start_date, end_date = st.date_input(
                "Rango de fechas a analizar",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )

            # Filtrar por el rango elegido
            mask = (
                (sdf["date"] >= pd.to_datetime(start_date)) &
                (sdf["date"] <= pd.to_datetime(end_date))
            )
            sdf_range = sdf.loc[mask].copy()

            if sdf_range.empty:
                st.info("No hay datos en el rango seleccionado.")
            else:
                # Gráfico de la serie filtrada
                st.line_chart(
                    sdf_range.set_index("date")[["y_true", "y_pred"]],
                    use_container_width=True,
                )
                st.caption(f"Archivo: {forecast_csv.split('/')[-1]}")

                # KPIs del rango
                total_real = float(sdf_range["y_true"].sum())
                total_pred = float(sdf_range["y_pred"].sum())
                diff = total_pred - total_real
                diff_pct = (diff / total_real * 100) if total_real != 0 else np.nan

                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Ventas reales (rango)",
                    f"{total_real:,.0f}".replace(",", " ")
                )
                c2.metric(
                    "Ventas pronosticadas (rango)",
                    f"{total_pred:,.0f}".replace(",", " ")
                )
                c3.metric(
                    "Diferencia % total",
                    f"{diff_pct:+.1f}%" if diff_pct == diff_pct else "—"
                )

                # Texto interpretativo dinámico
                diff_str = f"{diff:+,.0f}".replace(",", " ")

                st.markdown(
                    f"""
                    **Análisis operativo para la tienda {s_sel}, familia {f_sel}**

                    - Periodo analizado: **{start_date} → {end_date}**  
                    - El modelo estima **{total_pred:,.0f}** unidades/moneda en este rango.  
                    - Frente a las ventas reales (**{total_real:,.0f}**), la diferencia acumulada es de  
                      **{diff_str}** unidades (**{diff_pct:+.1f}%**).

                    Esto te muestra, en términos simples, el **margen de error esperado**
                    para esta combinación tienda–familia y periodo concreto.
                    """
                )
        else:
            st.info("No hay datos para esa combinación tienda–familia.")

    except Exception as e:
        st.error(f"No pude leer/mostrar forecast: {e}")
else:
    st.info("Aún no hay forecast_*.csv en reports/")
