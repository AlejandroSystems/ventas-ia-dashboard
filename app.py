import json
import re
import pandas as pd
import numpy as np
import streamlit as st
import fsspec

# ==========================================
# CONFIG BÁSICA DE LA APP
# ==========================================
st.set_page_config(
    page_title="Predicción de Ventas con IA – Grupo 01",
    layout="wide"
)
st.title("Sistema de Predicción de Ventas con IA – Grupo 01 (Universidad Autónoma del Perú)")

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
        ret
