# -------------------------------
# Carga de credenciales (2 modos)
# -------------------------------
def _normalize_bucket(b: str) -> str:
    b = (b or "").strip()
    if not b:
        return ""
    if not b.startswith("gs://"):
        b = "gs://" + b
    return b.rstrip("/")

def load_bucket_and_token_from_secrets():
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
        # usar .get para evitar KeyError y dar mensaje claro
        required = ["project_id","bucket","private_key_id","private_key","client_email",
                    "client_id","auth_uri","token_uri","auth_provider_x509_cert_url",
                    "client_x509_cert_url"]
        missing = [k for k in required if not gcp.get(k)]
        if missing:
            st.error(f"Faltan claves en [gcp] Secrets: {missing}")
            st.stop()
        bucket_b = _normalize_bucket(gcp.get("bucket"))
        sa_info = {
            "type": "service_account",
            "project_id": gcp.get("project_id"),
            "private_key_id": gcp.get("private_key_id"),
            "private_key": gcp.get("private_key"),  # con \n embebidos
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
