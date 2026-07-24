import io
import os
import re
import uuid
import unicodedata
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from pymongo import MongoClient
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="Comparador de Datasets", layout="wide")

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
MAX_COLUMNS = 7
MAX_ROWS_BUFFER = 20000
DATE_FORMATS = ["%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d"]
CSV_SEPARATORS = {"Coma (,)": ",", "Punto y coma (;)": ";", "Barra vertical (|)": "|", "Tabulador": "\t", "Auto-detectar": None}
DATE_DEFAULT = datetime(1950, 1, 1, tzinfo=timezone.utc)
NUMBER_DEFAULT = 0.0
STRING_DEFAULT = "(Default)"
TYPE_OPTIONS = ["String", "Date", "Number"]
NAME_MAXLEN = 11   # dataset display-name truncation
COL_MAXLEN = 7      # grouping/sum column-name truncation

# ─────────────────────────────────────────
# HELPER FUNCTIONS — naming
# ─────────────────────────────────────────
def dataset_label(dataset_key):
    """Filename (no extension), truncated to NAME_MAXLEN, always suffixed _A/_B."""
    filename = st.session_state.filenames.get(dataset_key) or f"Dataset{dataset_key}"
    base = os.path.splitext(filename)[0]
    if len(base) > NAME_MAXLEN:
        base = base[:NAME_MAXLEN]
    return f"{base}_{dataset_key}"


def short_col(name):
    """Column name truncated to COL_MAXLEN (no suffix — used for shared key columns)."""
    return name[:COL_MAXLEN] if len(name) > COL_MAXLEN else name


def sum_col_name(colname, dataset_key):
    """sum_<colname trunc to 7>_A / _B"""
    return f"sum_{short_col(colname)}_{dataset_key}"


def round_amount(x, decimals=4):
    """Round a numeric amount to a fixed number of decimal places, NaN-safe."""
    if x is None or pd.isna(x):
        return x
    return round(float(x), decimals)


DIFF_TOLERANCE = 0.0001  # differences below this are treated as matches, not flagged


def fmt_amount(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return f"{v:,.4f}"


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
            # Fallback: the file may carry the date in a different literal
            # shape than the chosen format (e.g. Excel auto-parsed it to
            # "2024-05-13 00:00:00"). Let pandas infer it, respecting the
            # day/month order implied by the format the user picked.
            dayfirst = fmt.startswith("%d")
            try:
                parsed = pd.to_datetime(text, dayfirst=dayfirst, errors="coerce")
                if not pd.isna(parsed):
                    dt = parsed.to_pydatetime()
            except Exception:
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
        return round(float(text), 4)
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
if "filenames" not in st.session_state:
    st.session_state.filenames = {"A": None, "B": None}

st.title("🔍 Comparador de Datasets (Streamlit + MongoDB buffer)")

# ─────────────────────────────────────────
# STEP 1 — UPLOAD
# ─────────────────────────────────────────
st.header("1️⃣ Carga de archivos")
up_col1, up_col2 = st.columns(2)


def load_raw(file, sheet_name=None, sep=None):
    if file.name.lower().endswith(".csv"):
        if sep is None:
            df = pd.read_csv(file, dtype=str, keep_default_na=False, sep=None, engine="python")
        else:
            df = pd.read_csv(file, dtype=str, keep_default_na=False, sep=sep)
    else:
        df = pd.read_excel(file, dtype=str, sheet_name=sheet_name if sheet_name is not None else 0)
    return df.astype(str)


def get_sheet_names(file):
    if file.name.lower().endswith((".xlsx", ".xls")):
        try:
            xls = pd.ExcelFile(file)
            file.seek(0)
            return xls.sheet_names
        except Exception:
            return None
    return None


with up_col1:
    file_a = st.file_uploader("Dataset A", type=["csv", "xlsx", "xls"], key="uploader_A")
    if file_a is not None:
        st.session_state.filenames["A"] = file_a.name
        if file_a.name.lower().endswith(".csv"):
            sep_label_a = st.selectbox("Separador CSV (A)", list(CSV_SEPARATORS.keys()), key="sep_A")
            file_a.seek(0)
            st.session_state.raw["A"] = load_raw(file_a, sep=CSV_SEPARATORS[sep_label_a])
        else:
            sheet_names_a = get_sheet_names(file_a)
            sheet_a = None
            if sheet_names_a:
                sheet_a = (
                    st.selectbox("Hoja de Excel (A)", sheet_names_a, key="sheet_A")
                    if len(sheet_names_a) > 1 else sheet_names_a[0]
                )
            file_a.seek(0)
            st.session_state.raw["A"] = load_raw(file_a, sheet_name=sheet_a)

with up_col2:
    file_b = st.file_uploader("Dataset B", type=["csv", "xlsx", "xls"], key="uploader_B")
    if file_b is not None:
        st.session_state.filenames["B"] = file_b.name
        if file_b.name.lower().endswith(".csv"):
            sep_label_b = st.selectbox("Separador CSV (B)", list(CSV_SEPARATORS.keys()), key="sep_B")
            file_b.seek(0)
            st.session_state.raw["B"] = load_raw(file_b, sep=CSV_SEPARATORS[sep_label_b])
        else:
            sheet_names_b = get_sheet_names(file_b)
            sheet_b = None
            if sheet_names_b:
                sheet_b = (
                    st.selectbox("Hoja de Excel (B)", sheet_names_b, key="sheet_B")
                    if len(sheet_names_b) > 1 else sheet_names_b[0]
                )
            file_b.seek(0)
            st.session_state.raw["B"] = load_raw(file_b, sheet_name=sheet_b)

st.caption("📄 Para archivos Excel con varias hojas, elige la hoja a usar; se carga solo una hoja por archivo. Para CSV, elige el separador o deja \"Auto-detectar\".")

if st.session_state.raw["A"] is None or st.session_state.raw["B"] is None:
    st.info("Sube ambos archivos para continuar.")
    st.stop()

# ─────────────────────────────────────────
# STEP 2 & 3 — COLUMN SELECTION + TYPE TAGGING
# ─────────────────────────────────────────
st.header("2️⃣ Selección de columnas (máx. 7) y tipo de dato")


def sniff_type_and_format(series, sample_size=50):
    """Heuristic sniffing (similar spirit to csv.Sniffer) to suggest a type/format for a column."""
    sample = series.dropna().astype(str).str.strip()
    sample = sample[sample != ""]
    sample = sample.head(sample_size)
    if sample.empty:
        return "String", None, None

    for fmt in DATE_FORMATS:
        matches = 0
        for v in sample:
            try:
                datetime.strptime(v, fmt)
                matches += 1
            except ValueError:
                pass
        if matches / len(sample) >= 0.8:
            return "Date", fmt, None

    numeric_pattern = re.compile(r"^-?[\d.,]+$")
    numeric_matches = sum(1 for v in sample if numeric_pattern.match(v))
    if numeric_matches / len(sample) >= 0.8:
        hint = None
        for v in sample:
            if "," in v and "." in v:
                hint = "1.234,56" if v.rfind(",") > v.rfind(".") else "1,234.56"
                break
        return "Number", None, hint

    return "String", None, None


def column_config_ui(dataset_key):
    df = st.session_state.raw[dataset_key]
    label = dataset_label(dataset_key)
    st.subheader(label)

    selected_cols = st.multiselect(
        f"Columnas a conservar ({label}) — máximo {MAX_COLUMNS}",
        options=list(df.columns),
        max_selections=MAX_COLUMNS,
        key=f"select_cols_{dataset_key}",
    )

    col_config = {}
    for col in selected_cols:
        st.markdown(f"**Columna: `{col}`**")

        suggested_type, suggested_fmt, suggested_num_hint = sniff_type_and_format(df[col])
        st.caption(
            f"🔎 Sugerencia detectada: **{suggested_type}**"
            + (f" (formato: `{suggested_fmt}`)" if suggested_fmt else "")
            + (f" (formato numérico: `{suggested_num_hint}`)" if suggested_num_hint else "")
        )

        col_type = st.radio(
            f"Tipo de dato para `{col}` ({dataset_key})",
            TYPE_OPTIONS,
            index=TYPE_OPTIONS.index(suggested_type),
            key=f"type_{dataset_key}_{col}",
            horizontal=True,
        )

        entry = {"type": col_type}

        if col_type == "Date":
            date_radio_options = DATE_FORMATS + ["Ninguno de los anteriores (personalizado)"]
            default_index = DATE_FORMATS.index(suggested_fmt) if suggested_fmt in DATE_FORMATS else len(DATE_FORMATS)
            fmt_choice = st.radio(
                f"Formato de fecha para `{col}`",
                date_radio_options,
                index=default_index,
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
                value=suggested_num_hint or "",
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


def build_summary(dataset_key, processed_df, original_col_count):
    config = st.session_state.config[dataset_key]
    summary = {
        "total_records": len(processed_df),
        "num_columns": original_col_count,
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
            original_col_count = st.session_state.raw[key].shape[1]
            st.session_state.summary[key] = build_summary(key, processed, original_col_count)
            push_to_buffer(key, processed.head(MAX_ROWS_BUFFER))
        st.success("Datos procesados y cargados al buffer.")

st.caption(f"⚠️ Esta versión limita la carga al buffer de MongoDB a las primeras {MAX_ROWS_BUFFER:,} filas y {MAX_COLUMNS} columnas por dataset. Las estadísticas de resumen (arriba) reflejan el dataset completo; la agrupación/comparación usa solo las filas cargadas al buffer.")

if st.session_state.summary["A"] and st.session_state.summary["B"]:
    sum_col1, sum_col2 = st.columns(2)
    for key, col in (("A", sum_col1), ("B", sum_col2)):
        summary = st.session_state.summary[key]
        with col:
            st.subheader(f"📊 Resumen — {dataset_label(key)}")
            st.metric("Total de registros", summary["total_records"])
            st.metric("Número de columnas", summary["num_columns"])
            if summary["number_sums"]:
                formatted_sums = {k: fmt_amount(v) for k, v in summary["number_sums"].items()}
                st.write("**Suma por columna numérica:**", formatted_sums)
            if summary["date_distinct_counts"]:
                st.write("**Conteo de fechas distintas:**", summary["date_distinct_counts"])
            if summary["string_distinct_counts"]:
                st.write("**Conteo de textos distintas:**", summary["string_distinct_counts"])

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
    st.markdown(f"**{dataset_label('A')}**")
    str_keys_a = date_keys_a = []
    if group_mode in ("Solo texto (strings)", "Ambos combinados"):
        str_keys_a = st.multiselect("Columnas de texto para agrupar (A)", cols_by_type("A", "String"), key="gstr_a")
    if group_mode in ("Solo fecha", "Ambos combinados"):
        date_keys_a = st.multiselect("Columna(s) de fecha para agrupar (A)", cols_by_type("A", "Date"), key="gdate_a", max_selections=1)
    num_col_a = st.selectbox("Columna numérica a sumar (A)", cols_by_type("A", "Number"), key="gnum_a")

with g2:
    st.markdown(f"**{dataset_label('B')}**")
    str_keys_b = date_keys_b = []
    if group_mode in ("Solo texto (strings)", "Ambos combinados"):
        str_keys_b = st.multiselect("Columnas de texto para agrupar (B)", cols_by_type("B", "String"), key="gstr_b")
    if group_mode in ("Solo fecha", "Ambos combinados"):
        date_keys_b = st.multiselect("Columna(s) de fecha para agrupar (B)", cols_by_type("B", "Date"), key="gdate_b", max_selections=1)
    num_col_b = st.selectbox("Columna numérica a sumar (B)", cols_by_type("B", "Number"), key="gnum_b")


def group_and_sum(dataset_key, str_keys, date_keys, num_col, key_names):
    df = read_from_buffer(dataset_key)
    group_cols = list(str_keys) + list(date_keys)
    grouped = df.groupby(group_cols, as_index=False)[num_col].sum()
    grouped[num_col] = grouped[num_col].apply(round_amount)
    sum_name = sum_col_name(num_col, dataset_key)
    grouped = grouped.rename(columns={num_col: sum_name})
    # key columns take dataset A's names (truncated), shared across A and B for alignment
    rename_map = {c: key_names[i] for i, c in enumerate(group_cols)}
    grouped = grouped.rename(columns=rename_map)
    key_cols = list(rename_map.values())
    grouped = grouped.sort_values(by=key_cols, ascending=True).reset_index(drop=True)
    return grouped, key_cols, sum_name


if st.button("📐 Agrupar y comparar"):
    keys_a = str_keys_a + date_keys_a
    keys_b = str_keys_b + date_keys_b
    if not keys_a or not keys_b:
        st.error("Selecciona al menos una columna de agrupación en cada dataset.")
    elif len(keys_a) != len(keys_b):
        st.error("El número de columnas de agrupación debe coincidir entre A y B para poder alinear.")
    else:
        key_names = [short_col(c) for c in keys_a]  # names come from dataset A
        grouped_a, key_cols, sum_name_a = group_and_sum("A", str_keys_a, date_keys_a, num_col_a, key_names)
        grouped_b, _, sum_name_b = group_and_sum("B", str_keys_b, date_keys_b, num_col_b, key_names)

        merged = pd.merge(
            grouped_a, grouped_b, on=key_cols, how="outer"
        ).sort_values(by=key_cols, ascending=True).reset_index(drop=True)

        merged["Diferencia"] = (merged[sum_name_a] - merged[sum_name_b]).apply(round_amount)
        st.session_state.comparison = {
            "df": merged, "key_cols": key_cols,
            "sum_a": sum_name_a, "sum_b": sum_name_b,
            "str_a": str_keys_a[0] if str_keys_a else None,
            "num_a": num_col_a,
            "str_b": str_keys_b[0] if str_keys_b else None,
            "num_b": num_col_b,
        }
        st.session_state.grouped = {"A": grouped_a, "B": grouped_b}

if st.session_state.comparison is not None:
    comp = st.session_state.comparison
    merged = comp["df"]
    sum_a, sum_b = comp["sum_a"], comp["sum_b"]

    def highlight_diff(row):
        styles = [""] * len(row)
        a, b = row.get(sum_a), row.get(sum_b)
        mismatch = pd.isna(a) or pd.isna(b) or abs(a - b) >= DIFF_TOLERANCE
        if mismatch:
            styles = ["background-color: orange"] * len(row)
        return styles

    st.subheader("🔀 Comparación lado a lado")
    styled = (
        merged.style
        .apply(highlight_diff, axis=1)
        .format({sum_a: fmt_amount, sum_b: fmt_amount, "Diferencia": fmt_amount})
    )
    st.dataframe(styled, use_container_width=True)

    # ─────────────────────────────────────────
    # STEP 6 — PDF EXPORT
    # ─────────────────────────────────────────
    def summary_table_flowable(styles):
        """Two-column (A | B) summary block, same layout as the on-screen view."""
        rows = [["Métrica", dataset_label("A"), dataset_label("B")]]
        sa, sb = st.session_state.summary["A"], st.session_state.summary["B"]
        rows.append(["Total de registros", sa["total_records"], sb["total_records"]])
        rows.append(["Número de columnas", sa["num_columns"], sb["num_columns"]])
        rows.append(["Suma por columna numérica",
                     str({k: fmt_amount(v) for k, v in sa["number_sums"].items()}),
                     str({k: fmt_amount(v) for k, v in sb["number_sums"].items()})])
        rows.append(["Fechas distintas", str(sa["date_distinct_counts"]), str(sb["date_distinct_counts"])])
        rows.append(["Textos distintos", str(sa["string_distinct_counts"]), str(sb["string_distinct_counts"])])
        table = Table(rows, repeatRows=1, colWidths=[150, 250, 250])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        return table

    def distribution_charts_side_by_side(chart_specs):
        """One combined image: dataset A and B bar charts side by side, with total labels on each bar."""
        valid_specs = [(k, s, n) for k, s, n in chart_specs if s and n]
        if not valid_specs:
            return None
        fig, axes = plt.subplots(1, len(valid_specs), figsize=(6 * len(valid_specs), 3.2))
        if len(valid_specs) == 1:
            axes = [axes]
        for ax, (key, str_col, num_col) in zip(axes, valid_specs):
            df = read_from_buffer(key)
            agg = df.groupby(str_col, as_index=False)[num_col].sum().sort_values(num_col, ascending=False).head(10)
            bars = ax.bar(agg[str_col].astype(str), agg[num_col], color="#4C72B0")
            labels = [f"{v:,.4f}" for v in agg[num_col]]
            ax.bar_label(bars, labels=labels, fontsize=6, rotation=0, padding=2)
            ax.set_title(f"{dataset_label(key)}: {num_col} por {str_col}", fontsize=9)
            ax.tick_params(axis="x", rotation=75, labelsize=6)
            ax.tick_params(axis="y", labelsize=7)
            ax.margins(y=0.15)
        fig.tight_layout()
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png", dpi=150)
        plt.close(fig)
        img_buf.seek(0)
        return Image(img_buf, width=350 * len(valid_specs), height=180)

    def build_pdf(df, sum_a, sum_b, chart_specs):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=landscape(letter))
        doc.author = "https://www.linkedin.com/in/asotov/"
        styles = getSampleStyleSheet()
        elements = [Paragraph("Comparación de Datasets", styles["Title"]), Spacer(1, 12)]

        elements.append(Paragraph("Resumen por dataset", styles["Heading2"]))
        elements.append(summary_table_flowable(styles))
        elements.append(Spacer(1, 16))

        elements.append(Paragraph("Distribución de montos", styles["Heading2"]))
        chart_img = distribution_charts_side_by_side(chart_specs)
        if chart_img is not None:
            elements.append(chart_img)
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Comparación agrupada", styles["Heading2"]))
        display_df = df.copy()
        for c in (sum_a, sum_b, "Diferencia"):
            if c in display_df.columns:
                display_df[c] = display_df[c].apply(fmt_amount)
        data = [list(display_df.columns)] + display_df.astype(str).values.tolist()
        table = Table(data, repeatRows=1)

        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
        ]
        for i, row in df.iterrows():
            a, b = row.get(sum_a), row.get(sum_b)
            mismatch = pd.isna(a) or pd.isna(b) or abs(a - b) >= DIFF_TOLERANCE
            if mismatch:
                style_cmds.append(("BACKGROUND", (0, i + 1), (-1, i + 1), colors.orange))

        table.setStyle(TableStyle(style_cmds))
        elements.append(table)
        doc.build(elements)
        buffer.seek(0)
        return buffer

    chart_specs = [
        ("A", comp["str_a"], comp["num_a"]),
        ("B", comp["str_b"], comp["num_b"]),
    ]
    pdf_buffer = build_pdf(merged, sum_a, sum_b, chart_specs)
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
    for key in ("raw", "config", "processed", "summary", "grouped", "filenames"):
        st.session_state[key] = {"A": None, "B": None} if key != "config" else {"A": {}, "B": {}}
    st.session_state.comparison = None
    st.success("Buffer eliminado. Puedes cargar nuevos archivos.")
    st.rerun()
