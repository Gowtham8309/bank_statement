import os
import io
import hashlib
import pandas as pd
import streamlit as st

# ---------- Inline defaults (real env vars override) ----------
os.environ.setdefault("VISION_AGENT_API_KEY",
    "d2U3aDNlMnY4NXF6aHVwc2JndmFsOlFGU2lVSEljRzJiRjZaNURqRGZMN1IzZmxZV09LdTls")
os.environ.setdefault("USE_AGENTIC_DOC", "1")
os.environ.setdefault("USE_GEMINI", "0")  # set "1" only if you want Gemini fallback
os.environ.setdefault("GOOGLE_API_KEY",
    "AIzaSyCXIN0sGxkhQmfJwFGrYYqb6IKRCU3WLwE")

from pipeline import process_pdf  # import AFTER env defaults

st.set_page_config(page_title="🏦 Bank Statement Extractor", page_icon="🏦", layout="wide")
st.title("🏦 Bank Statement Extractor")



# ---------- helpers ----------
DETAIL_COLS = [
    "AH_Name", "AH_Address", "AH_ContactNumber", "AH_Email",
    "BA_AccountNumber", "BA_IFSC", "BA_BranchAddress"
]
TXN_COLS = ["Date", "Description", "Ref", "Debit", "Credit", "Balance"]

def result_to_rows_with_details(result: dict, source_file: str) -> pd.DataFrame:
    """Flatten pipeline result into a CSV-friendly table that *includes* details on every row."""
    ah = result.get("account_holder", {}) or {}
    ba = result.get("bank_account", {}) or {}
    tx = result.get("transactions", []) or []

    def _clean(v):
        return "" if (v is None or str(v).strip().upper() == "N/A") else v

    details = {
        "AH_Name": _clean(ah.get("Name")),
        "AH_Address": _clean(ah.get("Address")),
        "AH_ContactNumber": _clean(ah.get("ContactNumber")),
        "AH_Email": _clean(ah.get("Email")),
        "BA_AccountNumber": _clean(ba.get("AccountNumber")),
        "BA_IFSC": _clean(ba.get("IFSC")),
        "BA_BranchAddress": _clean(ba.get("BranchAddress")),
    }

    rows = []
    if tx:
        for r in tx:
            row = {**details, "SourceFile": source_file}
            for c in TXN_COLS:
                row[c] = r.get(c)
            rows.append(row)
    else:
        row = {**details, "SourceFile": source_file}
        for c in TXN_COLS:
            row[c] = None
        rows.append(row)

    return pd.DataFrame(rows, columns=["SourceFile"] + DETAIL_COLS + TXN_COLS)

def _digest(name: str, data: bytes) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(name.encode("utf-8"))
    h.update(data)
    cfg = f"{bool(os.getenv('VISION_AGENT_API_KEY'))}|{os.getenv('USE_AGENTIC_DOC')}|{os.getenv('USE_GEMINI')}"
    h.update(cfg.encode("utf-8"))
    return h.hexdigest()

@st.cache_data(show_spinner=False, ttl=3600, max_entries=128)
def _process_bytes_cached(name: str, data: bytes) -> dict:
    """Cache wrapper that runs pipeline once for a given file content + config."""
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix=".pdf")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        return process_pdf(tmp)
    finally:
        try: os.close(fd)
        except: pass
        try: os.unlink(tmp)
        except: pass

# ---------- Multi-file upload + explicit Process (prevents loops) ----------
files = st.file_uploader(
    "Upload one or more **redacted** bank statement PDFs",
    type=["pdf"], accept_multiple_files=True
)
go = st.button("▶️ Process", type="primary", use_container_width=True)

# session store
if "processed_rows" not in st.session_state:
    st.session_state.processed_rows = {}   # digest -> DataFrame (with details)
if "provenance" not in st.session_state:
    st.session_state.provenance = {}       # digest -> provenance dict
if "file_names" not in st.session_state:
    st.session_state.file_names = {}       # digest -> original file name

if go and files:
    progress = st.progress(0.0)
    for idx, uploaded in enumerate(files, start=1):
        data = uploaded.getvalue()
        digest = _digest(uploaded.name, data)
        if digest not in st.session_state.processed_rows:
            with st.spinner(f"Processing {uploaded.name} ({idx}/{len(files)})…"):
                result = _process_bytes_cached(uploaded.name, data)
            df = result_to_rows_with_details(result, uploaded.name)
            st.session_state.processed_rows[digest] = df
            st.session_state.provenance[digest] = result.get("provenance", {})
            st.session_state.file_names[digest] = uploaded.name
        progress.progress(idx / len(files))

if not st.session_state.processed_rows:
    st.info("Upload PDFs and click **Process**.")
    st.stop()

# ---------- Show each processed file ----------
for digest, df in st.session_state.processed_rows.items():
    fname = st.session_state.file_names.get(digest, "statement.pdf")
    prov = st.session_state.provenance.get(digest, {})

    with st.expander(f"📄 {fname}", expanded=(len(st.session_state.processed_rows) == 1)):
        # top metrics
        prof = (prov or {}).get("doc_profile", {}) or {}
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pages", prof.get("pages", "—"))
        m2.metric("PDF Type", prof.get("type", "—"))
        m3.metric("DPI", prof.get("dpi", "—") or "—")
        m4.metric("Text Coverage", f"{prof.get('text_coverage', 0)}%")

        st.divider()

        # details cards (from first row)
        first = df.iloc[0].to_dict() if not df.empty else {}
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("👤 Account Holder")
            st.write(f"**Name:** {first.get('AH_Name') or 'N/A'}")
            st.write(f"**Address:** {first.get('AH_Address') or 'N/A'}")
            st.write(f"**Contact Number:** {first.get('AH_ContactNumber') or 'N/A'}")
            st.write(f"**Email:** {first.get('AH_Email') or 'N/A'}")
        with c2:
            st.subheader("🏦 Bank Account")
            st.write(f"**Account Number:** {first.get('BA_AccountNumber') or 'N/A'}")
            st.write(f"**IFSC:** {first.get('BA_IFSC') or 'N/A'}")
            st.write(f"**Branch Address:** {first.get('BA_BranchAddress') or 'N/A'}")

        st.divider()

        # transactions table
        st.subheader("📄 Transactions")
        txn_only = df[["SourceFile"] + TXN_COLS]
        if txn_only.dropna(how="all").empty:
            st.info("No transactions parsed.")
        else:
            # optional totals
            totals = {}
            if "Debit" in txn_only.columns:
                totals["Total Debit"] = pd.to_numeric(txn_only["Debit"], errors="coerce").dropna().sum()
            if "Credit" in txn_only.columns:
                totals["Total Credit"] = pd.to_numeric(txn_only["Credit"], errors="coerce").dropna().sum()
            if totals:
                t1, t2 = st.columns(2)
                t1.metric("Total Debit", f"{totals.get('Total Debit', 0):,.2f}")
                t2.metric("Total Credit", f"{totals.get('Total Credit', 0):,.2f}")

            st.dataframe(txn_only, use_container_width=True, height=420)

        # per-file downloads
        st.download_button(
            "⬇️ Download Transactions CSV",
            data=txn_only.to_csv(index=False).encode("utf-8"),
            file_name=f"{fname.rsplit('.',1)[0]}_transactions.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.download_button(
            "⬇️ Download CSV (with details on every row)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{fname.rsplit('.',1)[0]}_with_details.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # details-only single-row CSV
        details_only = df.iloc[[0]][[
            "AH_Name","AH_Address","AH_ContactNumber","AH_Email",
            "BA_AccountNumber","BA_IFSC","BA_BranchAddress"
        ]].copy()
        st.download_button(
            "⬇️ Download Details-only CSV",
            data=details_only.to_csv(index=False).encode("utf-8"),
            file_name=f"{fname.rsplit('.',1)[0]}_details_only.csv",
            mime="text/csv",
            use_container_width=True,
        )

        #st.subheader("ℹ️ Provenance & Metadata")
        #st.json(prov)

# ---------- Combined CSV ----------
st.markdown("---")
st.subheader("Bulk Export")
combined = pd.concat(list(st.session_state.processed_rows.values()), ignore_index=True)
st.download_button(
    "⬇️ Download **Combined CSV** (all files, with details)",
    data=combined.to_csv(index=False).encode("utf-8"),
    file_name="all_statements_with_details.csv",
    mime="text/csv",
    use_container_width=True,
)

# reset session
col_reset, _ = st.columns([1,2])
if col_reset.button("♻️ Reset"):
    st.session_state.clear()
    st.rerun()
