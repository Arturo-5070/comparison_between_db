import io
import re
import uuid
import unicodedata
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from pymongo import MongoClient
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="Comparador de Datasets", layout="wide")

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
MAX_COLUMNS = 7
DATE_FORMATS = ["%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d"]
DATE_DEFAULT = datetime(1950, 1, 1, tzinfo=timezone.utc)
NUMBER_DEFAULT = 0.0
STRING_DEFAULT = "(Default)"
TYPE_OPTIONS = ["String", "Date", "Number"]

# ─────────────────────────────────────────
# HELPER FUNCTIONS — cleaning / casting
# ─────────────────────────────────────────
def clean_string_value(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return STRING_DEFAULT
    text = str(value).strip()
    if text == "" or text.lower() in ("nan", "none", "nat"):
        return STRING_DEFAULT
    text = unicodedata.normalize("NFC", text)
    text = text.encode("utf-8", "ignore").decode("utf-8")
    return text


def parse_date_value(value, fmt=None, custom_regex=None):
    if value is None or str(value).strip() == "" or str(value).strip().lower() in ("nan", "none", "nat"):
        return DATE_DEFAULT
    text = str(value).strip()

    dt = None
    if fmt:
        try:
            dt = datetime.strptime(text, fmt)
        except ValueError:
            dt = None

    if dt is None and custom_regex:
        match = re.search(custom_regex, text)
        if match:
            try:
                captured = match.group(1) if match.groups() else match.group(0)
                # Best-effort: let pandas infer the captured fragment
                dt = pd.to_datetime(captured, errors="coerce")
                if pd.isna(dt):
                    dt = None
                else:
                    dt = dt.to_pydatetime()
            except Exception:
                dt = None

    if dt is None:
        return DATE_DEFAULT

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def parse_number_value(value, number_format=None):
    if value is None or str(value).strip() == "" or str(value).strip().lower() in ("nan", "none", "nat"):
        return NUMBER_DEFAULT
    text = str(value).strip()
    text = re.sub(r"[^\d,.\-]", "", text)

    if number_format:
        # number_format hint e.g. "1.234,56" (dot=thousands, comma=decimal)
        # or "1,234.56" (comma=thousands, dot=decimal)
        dot_pos = number_format.rfind(".")
        comma_pos = number_format.rfind(",")
        if dot_pos > comma_pos:
            # dot is decimal separator
            text = text.replace(".", "_DEC_").replace(",", "").replace("_DEC_", ".")
        elif comma_pos > dot_pos:
            # comma is decimal separator
            text = text.replace(",", "_DEC_").replace(".", "").replace("_DEC_", ".")
    else:
        # default assumption: comma = thousands, dot = decimal
        text = text.replace(",", "")

    try:
        return float(text)
    except ValueError:
        return NUMBER_DEFAULT


# ─────────────────────────────────────────
# CACHED MONGODB CONNECTION (buffer only — nothing persists)
# ─────────────────────────────────────────
@st.cache_resource
def get_client():
    uri = st.secrets["mongo"]["connection_string"]
    return MongoClient(uri)


def get_buffer_collection(dataset_key):
    client = get_client()
    db = client["dataset_compare_buffer"]
    return db[f"buf_{st.session_state.session_id}_{dataset_key}"]


def push_to_buffer(dataset_key, df):
    coll = get_buffer_collection(dataset_key)
    coll.delete_many({})
    if not df.empty:
        coll.insert_many(df.to_dict("records"))


def read_from_buffer(dataset_key):
    coll = get_buffer_collection(dataset_key)
    docs = list(coll.find({}, {"_id": 0}))
    return pd.DataFrame(docs)


def clear_all_buffers():
    for key in ("A", "B"):
        get_buffer_collection(key).drop()


# ─────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]
if "raw" not in st.session_state:
    st.session_state.raw = {"A": None, "B": None}
if "config" not in st.session_state:
    st.session_state.config = {"A": {}, "B": {}}
if "processed" not in st.session_state:
    st.session_state.processed = {"A": None, "B": None}
if "summary" not in st.session_state:
    st.session_state.summary = {"A": None, "B": None}
if "grouped" not in st.session_state:
    st.session_state.grouped = {"A": None, "B": None}
if "comparison" not in st.session_state:
    st.session_state.comparison = None

st.title("🔍 Comparador de Datasets (Streamlit + MongoDB buffer)")

# ─────────────────────────────────────────
# STEP 1 — UPLOAD
# ─────────────────────────────────────────
st.header("1️⃣ Carga de archivos")
up_col1, up_col2 = st.columns(2)


def load_raw(file):
    if file is None:
        return None
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(file, dtype=str, keep_default_na=False)
    else:
        df = pd.read_excel(file, dtype=str)
    return df.astype(str)


with up_col1:
    file_a = st.file_uploader("Dataset A", type=["csv", "xlsx", "xls"], key="uploader_A")
    if file_a is not None:
        st.session_state.raw["A"] = load_raw(file_a)

with up_col2:
    file_b = st.file_uploader("Dataset B", type=["csv", "xlsx", "xls"], key="uploader_B")
    if file_b is not None:
        st.session_state.raw["B"] = load_raw(file_b)

if st.session_state.raw["A"] is None or st.session_state.raw["B"] is None:
    st.info("Sube ambos archivos para continuar.")
    st.stop()

# ─────────────────────────────────────────
# STEP 2 & 3 — COLUMN SELECTION + TYPE TAGGING
# ─────────────────────────────────────────
st.header("2️⃣ Selección de columnas (máx. 7) y tipo de dato")


def column_config_ui(dataset_key):
    df = st.session_state.raw[dataset_key]
    st.subheader(f"Dataset {dataset_key}")

    selected_cols = st.multiselect(
        f"Columnas a conservar ({dataset_key}) — máximo {MAX_COLUMNS}",
        options=list(df.columns),
        max_selections=MAX_COLUMNS,
        key=f"select_cols_{dataset_key}",
    )

    col_config = {}
    for col in selected_cols:
        st.markdown(f"**Columna: `{col}`**")
        col_type = st.radio(
            f"Tipo de dato para `{col}` ({dataset_key})",
            TYPE_OPTIONS,
            key=f"type_{dataset_key}_{col}",
            horizontal=True,
        )

        entry = {"type": col_type}

        if col_type == "Date":
            fmt_choice = st.radio(
                f"Formato de fecha para `{col}`",
                DATE_FORMATS + ["Ninguno de los anteriores (personalizado)"],
                key=f"datefmt_{dataset_key}_{col}",
                horizontal=True,
            )
            if fmt_choice == "Ninguno de los anteriores (personalizado)":
                custom_regex = st.text_input(
                    f"Regex para extraer la fecha en `{col}`",
                    key=f"customregex_{dataset_key}_{col}",
                    placeholder=r"e.g. (\d{4}-\d{2}-\d{2})",
                )
                entry["fmt"] = None
                entry["regex"] = custom_regex
            else:
                entry["fmt"] = fmt_choice
                entry["regex"] = None

        elif col_type == "Number":
            convert = st.checkbox(
                f"Convertir `{col}` a número", value=True, key=f"convnum_{dataset_key}_{col}"
            )
            number_format = st.text_input(
                f"Formato de número de referencia para `{col}` (ej. 1,234.56 o 1.234,56)",
                key=f"numfmt_{dataset_key}_{col}",
            )
            entry["convert"] = convert
            entry["number_format"] = number_format

        col_config[col] = entry
        st.markdown("---")

    st.session_state.config[dataset_key] = col_config
    return selected_cols


cfg_col1, cfg_col2 = st.columns(2)
with cfg_col1:
    sel_a = column_config_ui("A")
with cfg_col2:
    sel_b = column_config_ui("B")

# ─────────────────────────────────────────
# STEP 4 — PROCESS: cast, null-fill, clean, push to Mongo buffer
# ─────────────────────────────────────────
st.header("3️⃣ Procesar y cargar al buffer")


def process_dataset(dataset_key, selected_cols):
    df = st.session_state.raw[dataset_key][selected_cols].copy()
    config = st.session_state.config[dataset_key]
    out = pd.DataFrame(index=df.index)

    for col in selected_cols:
        entry = config[col]
        if entry["type"] == "String":
            out[col] = df[col].apply(clean_string_value)
        elif entry["type"] == "Date":
            parsed = df[col].apply(
                lambda v: parse_date_value(v, fmt=entry.get("fmt"), custom_regex=entry.get("regex"))
            )
            out[col] = parsed.apply(lambda d: d.isoformat())
        elif entry["type"] == "Number":
            if entry.get("convert", True):
                out[col] = df[col].apply(lambda v: parse_number_value(v, entry.get("number_format")))
            else:
                out[col] = df[col].apply(clean_string_value)

    return out


def build_summary(dataset_key, processed_df):
    config = st.session_state.config[dataset_key]
    summary = {
        "total_records": len(processed_df),
        "num_columns": len(processed_df.columns),
        "number_sums": {},
        "date_distinct_counts": {},
        "string_distinct_counts": {},
    }
    for col, entry in config.items():
        if entry["type"] == "Number" and entry.get("convert", True):
            summary["number_sums"][col] = float(processed_df[col].sum())
        elif entry["type"] == "Date":
            summary["date_distinct_counts"][col] = processed_df[col].nunique()
        elif entry["type"] == "String":
            summary["string_distinct_counts"][col] = processed_df[col].nunique()
    return summary


if st.button("🚀 Procesar y cargar ambos datasets al buffer de MongoDB"):
    if not sel_a or not sel_b:
        st.error("Selecciona al menos una columna en cada dataset.")
    else:
        for key, sel in (("A", sel_a), ("B", sel_b)):
            processed = process_dataset(key, sel)
            st.session_state.processed[key] = processed
            push_to_buffer(key, processed)
            st.session_state.summary[key] = build_summary(key, processed)
        st.success("Datos procesados y cargados al buffer.")

for key in ("A", "B"):
    summary = st.session_state.summary[key]
    if summary:
        st.subheader(f"📊 Resumen — Dataset {key}")
        m1, m2 = st.columns(2)
        m1.metric("Total de registros", summary["total_records"])
        m2.metric("Número de columnas", summary["num_columns"])
        if summary["number_sums"]:
            st.write("**Suma por columna numérica:**", summary["number_sums"])
        if summary["date_distinct_counts"]:
            st.write("**Conteo de fechas distintas:**", summary["date_distinct_counts"])
        if summary["string_distinct_counts"]:
            st.write("**Conteo de textos distintos:**", summary["string_distinct_counts"])

if st.session_state.processed["A"] is None or st.session_state.processed["B"] is None:
    st.stop()

# ─────────────────────────────────────────
# STEP 5 — GROUPING
# ─────────────────────────────────────────
st.header("4️⃣ Agrupación y comparación")

group_mode = st.radio(
    "Modo de agrupación",
    ["Solo texto (strings)", "Solo fecha", "Ambos combinados"],
    horizontal=True,
)


def cols_by_type(dataset_key, col_type):
    return [c for c, e in st.session_state.config[dataset_key].items() if e["type"] == col_type]


g1, g2 = st.columns(2)

with g1:
    st.markdown("**Dataset A**")
    str_keys_a = date_keys_a = []
    if group_mode in ("Solo texto (strings)", "Ambos combinados"):
        str_keys_a = st.multiselect("Columnas de texto para agrupar (A)", cols_by_type("A", "String"), key="gstr_a")
    if group_mode in ("Solo fecha", "Ambos combinados"):
        date_keys_a = st.multiselect("Columna(s) de fecha para agrupar (A)", cols_by_type("A", "Date"), key="gdate_a", max_selections=1)
    num_col_a = st.selectbox("Columna numérica a sumar (A)", cols_by_type("A", "Number"), key="gnum_a")

with g2:
    st.markdown("**Dataset B**")
    str_keys_b = date_keys_b = []
    if group_mode in ("Solo texto (strings)", "Ambos combinados"):
        str_keys_b = st.multiselect("Columnas de texto para agrupar (B)", cols_by_type("B", "String"), key="gstr_b")
    if group_mode in ("Solo fecha", "Ambos combinados"):
        date_keys_b = st.multiselect("Columna(s) de fecha para agrupar (B)", cols_by_type("B", "Date"), key="gdate_b", max_selections=1)
    num_col_b = st.selectbox("Columna numérica a sumar (B)", cols_by_type("B", "Number"), key="gnum_b")


def group_and_sum(dataset_key, str_keys, date_keys, num_col):
    df = read_from_buffer(dataset_key)
    group_cols = list(str_keys) + list(date_keys)
    grouped = df.groupby(group_cols, as_index=False)[num_col].sum()
    grouped = grouped.rename(columns={num_col: "Sum"})
    # rename group columns to generic Key_n for cross-dataset alignment
    rename_map = {c: f"Key_{i+1}" for i, c in enumerate(group_cols)}
    grouped = grouped.rename(columns=rename_map)
    key_cols = list(rename_map.values())
    grouped = grouped.sort_values(by=key_cols, ascending=True).reset_index(drop=True)
    return grouped, key_cols


if st.button("📐 Agrupar y comparar"):
    keys_a = str_keys_a + date_keys_a
    keys_b = str_keys_b + date_keys_b
    if not keys_a or not keys_b:
        st.error("Selecciona al menos una columna de agrupación en cada dataset.")
    elif len(keys_a) != len(keys_b):
        st.error("El número de columnas de agrupación debe coincidir entre A y B para poder alinear.")
    else:
        grouped_a, key_cols = group_and_sum("A", str_keys_a, date_keys_a, num_col_a)
        grouped_b, _ = group_and_sum("B", str_keys_b, date_keys_b, num_col_b)

        merged = pd.merge(
            grouped_a, grouped_b, on=key_cols, how="outer", suffixes=("_A", "_B")
        ).sort_values(by=key_cols, ascending=True).reset_index(drop=True)

        merged["Diferencia"] = merged["Sum_A"] - merged["Sum_B"]
        st.session_state.comparison = {"df": merged, "key_cols": key_cols}
        st.session_state.grouped = {"A": grouped_a, "B": grouped_b}

if st.session_state.comparison is not None:
    merged = st.session_state.comparison["df"]

    def highlight_diff(row):
        styles = [""] * len(row)
        mismatch = pd.isna(row.get("Sum_A")) or pd.isna(row.get("Sum_B")) or row.get("Sum_A") != row.get("Sum_B")
        if mismatch:
            styles = ["background-color: orange"] * len(row)
        return styles

    st.subheader("🔀 Comparación lado a lado")
    st.dataframe(merged.style.apply(highlight_diff, axis=1), use_container_width=True)

    # ─────────────────────────────────────────
    # STEP 6 — PDF EXPORT
    # ─────────────────────────────────────────
    def build_pdf(df):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=landscape(letter))
        styles = getSampleStyleSheet()
        elements = [Paragraph("Comparación de Datasets", styles["Title"]), Spacer(1, 12)]

        data = [list(df.columns)] + df.astype(str).values.tolist()
        table = Table(data, repeatRows=1)

        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
        ]
        for i, row in df.iterrows():
            mismatch = pd.isna(row.get("Sum_A")) or pd.isna(row.get("Sum_B")) or row.get("Sum_A") != row.get("Sum_B")
            if mismatch:
                style_cmds.append(("BACKGROUND", (0, i + 1), (-1, i + 1), colors.orange))

        table.setStyle(TableStyle(style_cmds))
        elements.append(table)
        doc.build(elements)
        buffer.seek(0)
        return buffer

    pdf_buffer = build_pdf(merged)
    st.download_button(
        "📄 Descargar comparación en PDF",
        data=pdf_buffer,
        file_name="comparacion_datasets.pdf",
        mime="application/pdf",
    )

# ─────────────────────────────────────────
# STEP 7 — CLEAR BUFFER (MongoDB holds nothing permanently)
# ─────────────────────────────────────────
st.header("5️⃣ Finalizar")
if st.button("🗑️ Borrar buffer de MongoDB y reiniciar"):
    clear_all_buffers()
    for key in ("raw", "config", "processed", "summary", "grouped"):
        st.session_state[key] = {"A": None, "B": None} if key != "config" else {"A": {}, "B": {}}
    st.session_state.comparison = None
    st.success("Buffer eliminado. Puedes cargar nuevos archivos.")
    st.rerun()
