"""
Bank Statement Extraction Pipeline (fixed)
- Transactions:
    1) Vision Agent (agentic-doc)  → structured rows
    2) pdfplumber                  → tables
    3) HDFC-like text parser       → wrapped narrations
    4) Regex header split          → generic
    5) Gemini LLM fallback (STRICT JSON) → only if everything else fails
- Details:
    Agentic-Doc (guided schema) → Gemini JSON → Heuristic | field-wise merge + validation
"""

from __future__ import annotations

import os
import re
import json
import time
import logging
from typing import List, Dict, Optional, Tuple, Iterable, Any

import pandas as pd
from dateutil.parser import parse as dateparse

# ---------- Inline defaults (real env vars override) ----------
INLINE_ENV = {
    "VISION_AGENT_API_KEY": "d2U3aDNlMnY4NXF6aHVwc2JndmFsOlFGU2lVSEljRzJiRjZaNURqRGZMN1IzZmxZV09LdTls",
    "USE_AGENTIC_DOC": "1",
    "USE_VISION_AGENT_TX": "1",        # Vision Agent for transactions first
    "GOOGLE_API_KEY": "AIzaSyCXIN0sGxkhQmfJwFGrYYqb6IKRCU3WLwE",
    "GEMINI_MODEL": "gemini-2.5-pro",
    "USE_GEMINI": "1",
    "USE_LLM_TABLE_FALLBACK": "1",
}
for k, v in INLINE_ENV.items():
    os.environ.setdefault(k, v)

# ---------- PDF (no external binaries) ----------
try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None

# --- optional validators (do NOT crash if files are missing) ---
try:
    from validators_runner import validate_all_fields_and_merge
except Exception:
    validate_all_fields_and_merge = None  # type: ignore

try:
    from transactions_validator import validate_and_fix_transactions
except Exception:
    validate_and_fix_transactions = None  # type: ignore

# ---------- Optional Gemini ----------
try:
    import google.generativeai as genai
except Exception:
    genai = None

# ---------- Agentic-Doc ----------
try:
    from pydantic import BaseModel, Field
    from agentic_doc.parse import parse as agentic_parse
    _HAS_AGENTIC = True
except Exception:
    BaseModel = None  # type: ignore
    Field = None      # type: ignore
    agentic_parse = None
    _HAS_AGENTIC = False

# ---------- Optional pdfplumber ----------
try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except Exception:
    _HAS_PDFPLUMBER = False


# ======================= Helpers =======================

def _ifsc_ok(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.strip().upper()
    return s if re.match(r"^[A-Z]{4}0[A-Z0-9]{6}$", s) else None

def _clean_acct(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    digits = re.sub(r"\D", "", s)
    return digits if 9 <= len(digits) <= 18 else None

def _email_ok(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    return s.lower() if re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", s) else None

def _phone_prefer_contact(text: str) -> Optional[str]:
    """Prefer 10-digit Indian mobile; ignore 1800/helplines and weird counters."""
    if not text:
        return None
    # 1) Explicit 'contact|phone|mobile' labeled lines
    m = re.search(r"(?:contact|phone|mobile)[:\s]*\+?91?[-\s]?([6-9]\d{9})", text, flags=re.I)
    if m:
        return m.group(1)
    # 2) Any 10-13 digits -> pick 10-digit starting with 6-9; ignore 1800 etc.
    for m2 in re.finditer(r"\b(\+?91[-\s]?)?([6-9]\d{9})\b", text):
        cand = m2.group(2)
        if cand and not cand.startswith(("1800", "1860", "1008", "00000")):
            return cand
    return None

def _looks_like_date(s: str) -> bool:
    try:
        dateparse(s, dayfirst=True, fuzzy=True)
        return True
    except Exception:
        return False

def _to_iso_date(s: str) -> str:
    return dateparse(s, dayfirst=True, fuzzy=True).date().isoformat()

def _to_amount(s: Any) -> Optional[float]:
    if s is None:
        return None
    s = str(s).replace(",", "").replace("₹", "").strip()
    if not s:
        return None
    mul = -1.0 if s.upper().endswith("DR") else 1.0
    s = re.sub(r"(CR|DR)$", "", s, flags=re.I).strip()
    try:
        return float(s) * mul
    except Exception:
        return None

# ---------- Noise control for details ----------
# ---------- Noise guards & label utilities (replace your current versions) ----------

_NOISE_TOKENS = {
    "address": [
        "drawing power", "ifs code", "ifsc code", "overdraft", "account description",
        "balance as on", "search for", "statement of account", "page no", "generated on",
        "gstin", "gstn", "customer id", "cust id", "account status", "a/c open date",
        "micr", "branch code", "cr count", "dr count", "debits", "credits",
        "closing bal", "closing balance", "opening balance", "total withdrawals", "total deposits"
    ],
    "branch": [
        "account number", "a/c no", "date", "statement", "page no", "generated on", "ifs code", "ifsc"
    ],
    "name_bad": [
        "count", "credits", "debits", "closing", "balance", "account", "number",
        "date", "address", "summary", "totals"
    ]
}

_LABEL_PAT = re.compile(
    r"^\s*(name|address|contact|contact number|phone|mobile|email|e-mail|account\s*no|account number|a/c\s*no|ifsc|branch( address)?)\s*[:\-]\s*",
    re.I
)

def _strip_noise_tokens(text: Optional[str], kind: str) -> Optional[str]:
    if not text:
        return None
    s = " ".join(str(text).split())  # collapse whitespace
    for t in _NOISE_TOKENS.get(kind, []):
        s = re.sub(rf"\b{re.escape(t)}\b", "", s, flags=re.I)
    s = re.sub(r"\s{2,}", " ", s).strip(" ,;-:\n\t")
    return s or None

def _clean_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    s = str(name).strip()
    s = re.sub(r"^[\.\-:•\s]+", "", s)          # drop leading punctuation
    s = re.sub(r"[\.,:;•\-\s]+$", "", s)        # drop trailing punctuation
    s = re.sub(r"^(mr|mrs|ms|miss|smt|dr)\.?\s+", "", s, flags=re.I)
    s = re.sub(r"[^A-Za-z\s\.'-]", " ", s)      # letters + name chars
    s = re.sub(r"\s{2,}", " ", s).strip()
    if not s:
        return None
    # kill one-word junk like "Count"
    if s.lower() in _NOISE_TOKENS["name_bad"]:
        return None
    # must have at least one 2+ letter word
    if not any(len(w) >= 2 for w in s.split()):
        return None
    return s.title()

def _clean_address_like(s: Optional[str]) -> Optional[str]:
    """Remove labels, metrics, and numeric-only tokens from address-like fields."""
    s = _strip_noise_tokens(s, "address")
    if not s:
        return None
    # drop embedded 'Address :' labels and metric fragments
    s = re.sub(_LABEL_PAT, "", s).strip()
    parts = [p.strip() for p in re.split(r"[,\n;]+", s) if p.strip()]
    keep: list[str] = []
    metric_re = re.compile(r"(cr count|dr count|debits|credits|closing|balance)", re.I)
    for p in parts:
        if metric_re.search(p):
            continue
        # keep segments that contain letters (addresses) — drop those that are mostly numeric like "2,620.27" or "5"
        letters = sum(ch.isalpha() for ch in p)
        digits  = sum(ch.isdigit() for ch in p)
        if letters == 0 and digits > 0:
            continue
        keep.append(p)
    out = ", ".join(keep).strip(" ,;-")
    return out or None

def _clean_branch_addr(s: Optional[str]) -> Optional[str]:
    s = _strip_noise_tokens(s, "branch")
    if not s:
        return None
    s = re.sub(_LABEL_PAT, "", s).strip()
    # remove stray repeated "Address :" inside the value and collapse double colons
    s = re.sub(r"\bAddress\s*:\s*", "", s, flags=re.I)
    s = re.sub(r":\s*:", ":", s)
    s = re.sub(r"\s{2,}", " ", s).strip(" ,;-")
    return s or None

def _extract_first_match(pattern: str, text: str, group: int = 0) -> Optional[str]:
    m = re.search(pattern, text, flags=re.I)
    return m.group(group) if m else None

def _kv_pick(lines: list[str], key: str) -> Optional[str]:
    for ln in lines:
        if re.match(rf"^\s*{re.escape(key)}\s*[:\-]\s*", ln, flags=re.I):
            return re.sub(_LABEL_PAT, "", ln).strip()
    return None

def _derive_name_from_email(email: Optional[str]) -> Optional[str]:
    e = _email_ok(email)
    if not e:
        return None
    local = e.split("@", 1)[0]
    # split on non-letters and re-title
    tokens = [t for t in re.split(r"[^a-zA-Z]+", local) if len(t) >= 2]
    if len(tokens) >= 2:
        name = " ".join(tokens).title()
        # guard against generic usernames
        if name.lower() not in _NOISE_TOKENS["name_bad"]:
            return name
    return None



# ======================= Stage 1: Intake =======================

def validate_pdf(path: str) -> None:
    if fitz is None:
        raise RuntimeError("PyMuPDF (pymupdf) is required. Install `pymupdf`.")
    try:
        with fitz.open(path) as doc:
            if doc.is_encrypted:
                raise ValueError("PDF is encrypted. Export an unlocked copy.")
            if len(doc) == 0:
                raise ValueError("PDF has zero pages.")
    except Exception as e:
        raise ValueError(f"Failed to open PDF: {e}")

def extract_page_texts(path: str) -> List[str]:
    texts: List[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            texts.append(page.get_text("text") or "")
    return texts

def compute_doc_profile(path: str) -> Dict[str, Any]:
    with fitz.open(path) as doc:
        pages = len(doc)
        page_texts = [p.get_text("text") or "" for p in doc]
    total_chars = sum(len(t) for t in page_texts)
    avg_chars = total_chars / max(1, pages)
    doc_type = "text" if avg_chars >= 100 else "image"
    return {
        "pages": pages,
        "type": doc_type,
        "text_coverage": round(100.0 if avg_chars >= 100 else max(0.0, min(100.0, avg_chars)), 2),
        "dpi": None,
    }


# ======================= Stage 2: Transactions (algorithmic-first) =======================

# ---- 2.0 Vision Agent table extraction ----
if _HAS_AGENTIC:
    class _TxnRow(BaseModel):  # type: ignore[misc]
        date: Optional[str] = Field(
            default=None,
            description="Transaction date only (e.g., 12/08/2025 or 12-08-2025). Do NOT include time.",
        )
        description: Optional[str] = Field(
            default=None,
            description="Narration/description. Merge wrapped lines. No headers.",
        )
        ref: Optional[str] = Field(default=None, description="Reference/Cheque/UTR number if present.")
        debit: Optional[str] = Field(
            default=None,
            description="Withdrawal amount as a plain number with two decimals (e.g., 2500.00). No currency or commas. Empty if credit.",
        )
        credit: Optional[str] = Field(
            default=None,
            description="Deposit/credit amount as a plain number with two decimals. Empty if debit.",
        )
        balance: Optional[str] = Field(
            default=None,
            description="Running balance after the transaction as a plain number with two decimals.",
        )

    class _TxnOut(BaseModel):  # type: ignore[misc]
        transactions: List[_TxnRow] = Field(
            default_factory=list,
            description=(
                "All transaction rows in order as they appear in the bank statement. "
                "Skip headers and blank lines. Use 'debit' for withdrawals and 'credit' for deposits. "
                "If a field is missing, leave it blank."
            ),
        )
else:
    class _TxnOut(object):  # dummy
        pass

def _agentic_transactions_ok(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    good = (df["Date"].notna().sum() + df["Description"].notna().sum()) >= max(3, len(df) // 2)
    return bool(good)

def extract_transactions_with_agentic_doc(pdf_path: str) -> Tuple[pd.DataFrame, Dict[str, Any], bool]:
    if not _HAS_AGENTIC or agentic_parse is None:
        return pd.DataFrame(), {"anomalies": ["agentic-doc not available"]}, False

    pages = agentic_parse(pdf_path, extraction_model=_TxnOut)
    rows: List[Dict[str, Optional[str]]] = []
    for p in pages:
        e = getattr(p, "extraction", None)
        if not e:
            continue
        tx = getattr(e, "transactions", []) or []
        for r in tx:
            rows.append({
                "Date": getattr(r, "date", None),
                "Description": getattr(r, "description", None),
                "Ref": getattr(r, "ref", None),
                "Debit": getattr(r, "debit", None),
                "Credit": getattr(r, "credit", None),
                "Balance": getattr(r, "balance", None),
            })

    df = pd.DataFrame(rows, columns=["Date", "Description", "Ref", "Debit", "Credit", "Balance"]).dropna(how="all")
    if df.empty:
        return df, {"coverage_hits": 0, "anomalies": ["no rows from Vision Agent"]}, False

    # Normalize
    if "Date" in df.columns:
        def _norm_date(x):
            if not x:
                return None
            x = str(x).strip().replace(".", ":")
            x = re.split(r"\s+", x)[0]
            return _to_iso_date(x) if _looks_like_date(x) else None
        df["Date"] = df["Date"].apply(_norm_date)

    for col in ["Debit", "Credit", "Balance"]:
        if col in df.columns:
            df[col] = df[col].apply(_to_amount)

    if not _agentic_transactions_ok(df):
        return df, {"coverage_hits": len(df), "anomalies": ["Vision Agent rows below quality threshold"]}, False

    report = {
        "coverage_hits": int(len(df)),
        "anomalies": [],
        "pages_spanned": [],
        "algorithmic_success": True,
    }
    return df, report, True

# ---- 2.1 pdfplumber (structure-aware) ----
def _parse_transactions_pdfplumber(pdf_path: str) -> Tuple[pd.DataFrame, Dict[str, Any], bool]:
    if not _HAS_PDFPLUMBER:
        return pd.DataFrame(), {"anomalies": ["pdfplumber not available"]}, False
    rows_all: list[dict] = []
    pages_spanned: list[int] = []
    with pdfplumber.open(pdf_path) as pdf:
        for pidx, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables(
                table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "intersection_tolerance": 5,
                    "join_tolerance": 3,
                    "snap_tolerance": 3,
                    "min_words_vertical": 1,
                    "min_words_horizontal": 1,
                    "keep_blank_chars": False,
                }
            ) or []
            for tbl in tables:
                if not tbl or len(tbl) < 2:
                    continue
                header = [ (h or "").strip().lower() for h in tbl[0] ]
                def col_idx(names):
                    for i,h in enumerate(header):
                        for n in names:
                            if n in h: return i
                    return -1
                i_date = col_idx(["date", "txn date", "value date"])
                i_desc = col_idx(["description", "narration", "details", "particular"])
                i_ref  = col_idx(["ref", "cheque", "utr", "chq"])
                i_deb  = col_idx(["debit", "withdrawal", "dr"])
                i_cre  = col_idx(["credit", "deposit", "cr"])
                i_bal  = col_idx(["balance", "bal", "running balance"])
                if i_date < 0 or i_desc < 0 or (i_deb < 0 and i_cre < 0):
                    continue
                for r in tbl[1:]:
                    d = (r[i_date] or "").strip() if 0 <= i_date < len(r) else ""
                    desc = (r[i_desc] or "").strip() if 0 <= i_desc < len(r) else ""
                    if not d or not desc or not _looks_like_date(d):
                        continue
                    rows_all.append({
                        "Date": _to_iso_date(d),
                        "Description": desc,
                        "Ref": (r[i_ref] or "").strip() if 0 <= i_ref < len(r) and r[i_ref] else None,
                        "Debit": _to_amount(r[i_deb]) if 0 <= i_deb < len(r) else None,
                        "Credit": _to_amount(r[i_cre]) if 0 <= i_cre < len(r) else None,
                        "Balance": _to_amount(r[i_bal]) if 0 <= i_bal < len(r) else None,
                    })
                    pages_spanned.append(pidx)
    df = pd.DataFrame(rows_all)
    if df.empty:
        return df, {"coverage_hits": 0, "anomalies": ["no tables recognized"]}, False
    report = {
        "coverage_hits": len(df),
        "total_candidates": len(df),
        "pages_spanned": sorted(set(pages_spanned)),
        "algorithmic_success": len(df) >= 5,
        "anomalies": [],
    }
    return df, report, report["algorithmic_success"]

# ---- 2.2 HDFC-text parser (wrapped narration) ----
def _parse_transactions_hdfc_like(page_texts: Iterable[str]) -> Tuple[pd.DataFrame, Dict[str, Any], bool]:
    raw_lines: List[str] = []
    for page in page_texts:
        for ln in page.splitlines():
            ln = ln.replace("\t", " ").strip()
            if not ln:
                continue
            if any(k in ln for k in [
                "Page No .:", "Statement of account", "Registered Office Address",
                "GSTN", "GSTIN", "Generated On:", "Account Branch", "Nomination",
                "MICR", "IFSC", "A/C Open Date", "Account Status", "Branch Code",
            ]):
                continue
            ln = re.sub(r"\s{2,}", "  ", ln)
            raw_lines.append(ln)

    hdr_idx = None
    header_re = re.compile(r"\bDate\b.*\bNarration\b.*\bBalance\b", re.I)
    for i, l in enumerate(raw_lines):
        if header_re.search(l):
            hdr_idx = i
            break
    if hdr_idx is None:
        return pd.DataFrame(columns=["Date","Description","Ref","Debit","Credit","Balance"]), {
            "coverage_hits": 0, "total_candidates": 0, "pages_spanned": [],
            "algorithmic_success": False, "anomalies": ["HDFC header not found"]
        }, False

    work = raw_lines[hdr_idx + 1 :]
    is_date_start = re.compile(r"^\d{2}/\d{2}/\d{2}\b")

    merged_rows: List[str] = []
    buf = ""
    for ln in work:
        if is_date_start.match(ln):
            if buf:
                merged_rows.append(buf.strip())
            buf = ln
        else:
            if buf:
                buf += " " + ln
    if buf:
        merged_rows.append(buf.strip())

    money = r"(?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2}"
    right3 = re.compile(rf"(?P<w>{money})\s+(?P<d>{money})\s+(?P<b>{money})\s*$")
    right2 = re.compile(rf"(?P<d>{money})\s+(?P<b>{money})\s*$")
    right1 = re.compile(rf"(?P<b>{money})\s*$")

    out: List[Dict[str, Any]] = []
    for raw in merged_rows:
        wd = dep = bal = None
        m = right3.search(raw) or right2.search(raw) or right1.search(raw)
        left = raw
        if m:
            gd = m.groupdict()
            if "w" in gd and gd["w"]: wd = _to_amount(gd["w"])
            if "d" in gd and gd["d"]: dep = _to_amount(gd["d"])
            if "b" in gd and gd["b"]: bal = _to_amount(gd["b"])
            left = raw[: m.start()].rstrip()

        mdate = is_date_start.match(left)
        if not mdate:
            continue
        txn_date_str = mdate.group(0)
        rem = left[mdate.end():].strip()

        m_valdt = re.search(r"(\d{2}/\d{2}/\d{2})(?!.*\d{2}/\d{2}/\d{2})", rem)
        rem2 = rem[: m_valdt.start()].rstrip() if m_valdt else rem
        m_ref = re.search(r"(\d{8,})\s*$", rem2)
        ref = m_ref.group(1) if m_ref else None
        narr = rem2[: m_ref.start()].strip() if m_ref else rem2

        out.append({
            "Date": _to_iso_date(txn_date_str),
            "Description": narr if narr else None,
            "Ref": ref,
            "Debit": wd,
            "Credit": dep,
            "Balance": bal,
        })

    df = pd.DataFrame(out, columns=["Date","Description","Ref","Debit","Credit","Balance"])
    if df.empty:
        return df, {"coverage_hits": 0, "total_candidates": 0, "pages_spanned": [],
                    "algorithmic_success": False, "anomalies": ["HDFC rows not recognized"]}, False
    report = {"coverage_hits": len(df), "total_candidates": len(df), "pages_spanned": [],
              "algorithmic_success": len(df) >= 5, "anomalies": []}
    return df, report, report["algorithmic_success"]

# ---- 2.3 legacy regex fallback ----
def _parse_transactions_regex(page_texts: Iterable[str]) -> Tuple[pd.DataFrame, Dict[str, Any], bool]:
    header_line = None
    header_page_index = 0
    header_pattern = re.compile(r"(date|narration|description|debit|credit|balance)", re.I)

    for pidx, text in enumerate(page_texts):
        for line in text.splitlines():
            if header_pattern.search(line):
                header_line = line
                header_page_index = pidx
                break
        if header_line:
            break

    rows: List[List[str]] = []
    pages_spanned: List[int] = []
    if header_line:
        split_positions = [m.start() for m in re.finditer(r"\s{2,}", header_line)]
        col_positions = [0]
        for pos in split_positions:
            if pos - col_positions[-1] >= 2:
                col_positions.append(pos)
        col_positions.append(len(header_line))

        header_cols: List[str] = []
        for i in range(len(col_positions) - 1):
            seg = header_line[col_positions[i]:col_positions[i + 1]].strip()
            header_cols.append(seg or f"Column{i+1}")

        for pidx in range(header_page_index, len(page_texts)):
            for line in page_texts[pidx].splitlines():
                if len(line) < 10 or line.count(" ") < 2:
                    continue
                fields: List[str] = []
                for i in range(len(col_positions) - 1):
                    seg = line[col_positions[i]:col_positions[i + 1]].strip()
                    fields.append(seg)
                if fields and (_looks_like_date(fields[0]) or re.search(r"\d", " ".join(fields[-3:]))):
                    rows.append(fields)
                    pages_spanned.append(pidx + 1)

        if rows:
            df = pd.DataFrame(rows)
            if len(header_cols) < df.shape[1]:
                header_cols += [f"Column{i+1}" for i in range(len(header_cols), df.shape[1])]
            df.columns = header_cols[: df.shape[1]]

            rename_map = {}
            for c in df.columns:
                lc = c.lower()
                if "date" in lc and "value" not in lc:
                    rename_map[c] = "Date"
                elif any(k in lc for k in ["narration", "descr", "particular"]):
                    rename_map[c] = "Description"
                elif "ref" in lc or "chq" in lc or "utr" in lc:
                    rename_map[c] = "Ref"
                elif any(k in lc for k in ["debit", "withdrawal", "dr"]):
                    rename_map[c] = "Debit"
                elif any(k in lc for k in ["credit", "deposit", "cr"]):
                    rename_map[c] = "Credit"
                elif "balance" in lc or "bal" in lc:
                    rename_map[c] = "Balance"
            df = df.rename(columns=rename_map)
            df = df.loc[:, ~df.columns.duplicated()]

            if "Date" in df.columns:
                df["Date"] = df["Date"].apply(lambda x: _to_iso_date(x) if x and _looks_like_date(str(x)) else None)
            if "Debit" in df.columns:
                df["Debit"] = df["Debit"].apply(_to_amount)
            if "Credit" in df.columns:
                df["Credit"] = df["Credit"].apply(_to_amount)
            if "Balance" in df.columns:
                df["Balance"] = df["Balance"].apply(_to_amount)

            coverage_hits = len(df.index)
            algorithmic_success = coverage_hits >= 5
            report = {
                "coverage_hits": coverage_hits,
                "total_candidates": len(rows),
                "pages_spanned": sorted(set(pages_spanned)),
                "algorithmic_success": algorithmic_success,
                "anomalies": [] if algorithmic_success else [f"Unexpected number of columns detected: {df.shape[1]}"],
            }
            return df, report, algorithmic_success

    empty = pd.DataFrame(columns=["Date", "Description", "Ref", "Debit", "Credit", "Balance"])
    report = {"coverage_hits": 0, "total_candidates": 0, "pages_spanned": [], "algorithmic_success": False, "anomalies": ["No table header found"]}
    return empty, report, False


# ======================= Stage 2b: LLM fallback for tables (Gemini) =======================

def _surface_table_like_regions(page_texts: List[str]) -> str:
    """Grab lines near headers and following content; strip summary rows like 'Cr Count'."""
    buf: List[str] = []
    header_re = re.compile(r"(date|narration|description|debit|credit|balance)", re.I)
    skip_re = re.compile(r"(cr count|debits|credits|closing bal|closing balance|totals?)", re.I)
    for text in page_texts[:3]:  # first 3 pages are enough for LLM fallback
        lines = text.splitlines()
        capture = False
        for ln in lines:
            if header_re.search(ln):
                capture = True
                buf.append(ln)
                continue
            if capture:
                if skip_re.search(ln):
                    continue
                buf.append(ln)
                # stop after a big block to keep prompt small
                if len(buf) > 400:
                    break
    return "\n".join(buf[:1200])

def _gemini_json_call(system_prompt: str, user_prompt: str, model_name: Optional[str] = None, retries: int = 2) -> str:
    if os.getenv("USE_GEMINI", "0") != "1" or genai is None:
        return "{}"
    api_key = 'AIzaSyCXIN0sGxkhQmfJwFGrYYqb6IKRCU3WLwE'
    if not api_key:
        return "{}"
    model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    try:
        genai.configure(api_key=api_key)
        # Try with system_instruction & JSON mime
        try:
            model = genai.GenerativeModel(
                model_name,
                system_instruction=system_prompt,
                generation_config={"response_mime_type": "application/json"}
            )
        except Exception:
            model = genai.GenerativeModel(model_name)
        for _ in range(retries):
            try:
                resp = model.generate_content(user_prompt)
                text = (getattr(resp, "text", None) or "").strip()
                if text.startswith("{") or text.startswith("["):
                    return text
            except Exception as e:
                logging.warning(f"Gemini generate_content error: {e}")
                time.sleep(0.8)
        return "{}"
    except Exception as e:
        logging.warning(f"Gemini call failed: {e}")
        return "{}"

def extract_transactions_with_llm(page_texts: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any], bool]:
    """LLM fallback ONLY if algorithmic failed."""
    table_blob = _surface_table_like_regions(page_texts)
    if not table_blob or len(table_blob) < 40:
        return pd.DataFrame(), {"anomalies": ["no table-like text for LLM"]}, False

    system = (
        "You convert raw bank-statement rows into STRICT JSON. "
        "Skip headers, totals, and summaries like 'Cr Count', 'Debits', 'Credits', 'Closing Bal'. "
        "Do not invent values."
    )
    schema = {
        "transactions": [
            {
                "Date": "YYYY-MM-DD (ISO date; parse dd/mm/yy or dd-mm-yyyy, etc.)",
                "Description": "string",
                "Ref": "string or null",
                "Debit": "number or null (withdrawals); never with 'DR/CR'",
                "Credit": "number or null (deposits); never with 'DR/CR'",
                "Balance": "number or null"
            }
        ]
    }
    user = (
        "Return ONLY a JSON object with key 'transactions' exactly like this schema:\n"
        + json.dumps(schema, indent=2)
        + "\nRules:\n"
          "- One of Debit or Credit must be null for each row.\n"
          "- Combine wrapped descriptions into one line.\n"
          "- Ignore non-row text.\n"
          "- No extra keys, no prose.\n\n"
        "RAW TABLE TEXT:\n"
        f"{table_blob}"
    )
    out = _gemini_json_call(system, user)
    try:
        data = json.loads(out)
        rows = data.get("transactions", [])
    except Exception:
        return pd.DataFrame(), {"anomalies": ["LLM invalid JSON"]}, False

    parsed: List[Dict[str, Any]] = []
    for r in rows:
        d = r.get("Date")
        desc = r.get("Description")
        ref = r.get("Ref")
        debit = r.get("Debit")
        credit = r.get("Credit")
        balance = r.get("Balance")
        # normalize
        d_iso = _to_iso_date(d) if d and _looks_like_date(str(d)) else None
        parsed.append({
            "Date": d_iso,
            "Description": (desc or None),
            "Ref": (ref or None),
            "Debit": _to_amount(debit),
            "Credit": _to_amount(credit),
            "Balance": _to_amount(balance),
        })

    df = pd.DataFrame(parsed, columns=["Date","Description","Ref","Debit","Credit","Balance"]).dropna(how="all")
    df = df[df["Date"].notna() & df["Description"].notna()]
    ok = len(df) >= 5
    report = {"coverage_hits": len(df), "anomalies": [] if ok else ["too few rows"], "algorithmic_success": ok, "llm_used_for_tables": True}
    return df, report, ok


# ======================= Stage 3: Details =======================

if _HAS_AGENTIC:
    class _BankFields(BaseModel):  # type: ignore[misc]
        name: Optional[str] = Field(
            default=None,
            description="Account holder full name exactly as printed (no titles like Mr/Ms, no labels)."
        )
        address: Optional[str] = Field(
            default=None,
            description="Postal address lines for the account holder only. Exclude banking terms (Drawing Power, IFSC, Overdraft, etc.)."
        )
        contact_number: Optional[str] = Field(
            default=None,
            description="Primary phone number for the holder. Return 10 digits only if Indian."
        )
        email: Optional[str] = Field(
            default=None,
            description="Email address of the account holder (if present)."
        )
        account_number: Optional[str] = Field(
            default=None,
            description="Account number with 9–18 digits (no spaces)."
        )
        ifsc: Optional[str] = Field(
            default=None,
            description="IFSC in format 4 letters + 0 + 6 alphanumerics, e.g., HDFC0001234."
        )
        branch_address: Optional[str] = Field(
            default=None,
            description="Name/address of the branch only. Exclude labels like 'Account Number' or 'Date'."
        )
else:
    class _BankFields(object):
        pass

def extract_details_with_agentic_doc(pdf_path: str) -> Tuple[Dict[str, Optional[str]], Dict[str, Optional[str]], Dict]:
    if not _HAS_AGENTIC or agentic_parse is None:
        raise RuntimeError("agentic-doc not installed")
    try:
        pages = agentic_parse(pdf_path, extraction_model=_BankFields, pages=[0, 1])
    except TypeError:
        pages = agentic_parse(pdf_path, extraction_model=_BankFields)

    e = getattr(pages[0], "extraction", None)
    meta = getattr(pages[0], "extraction_metadata", None)

    raw_name   = getattr(e, "name", None) if e else None
    raw_addr   = getattr(e, "address", None) if e else None
    raw_phone  = getattr(e, "contact_number", None) if e else None
    raw_email  = getattr(e, "email", None) if e else None
    raw_acct   = getattr(e, "account_number", None) if e else None
    raw_ifsc   = getattr(e, "ifsc", None) if e else None
    raw_branch = getattr(e, "branch_address", None) if e else None

    try:
        full_text = "\n".join((p.markdown or "") for p in pages)
    except Exception:
        full_text = ""

    name  = _clean_name(raw_name) or _derive_name_from_email(raw_email) or _derive_name_from_email(_email_ok(raw_email))
    addr  = _clean_address_like(raw_addr)
    email = _email_ok(raw_email)
    phone = _phone_prefer_contact(raw_phone or "") or _phone_prefer_contact(full_text or "")
    acct  = _clean_acct(raw_acct) or _clean_acct(full_text or "")
    ifsc  = _ifsc_ok(raw_ifsc) or _ifsc_ok(full_text or "")
    branch= _clean_branch_addr(raw_branch)

    # If address is still too short, try header lines under the name
    if (not addr or len(addr) < 8):
        try:
            with fitz.open(pdf_path) as _doc:
                first_txt = _doc[0].get_text("text") or ""
        except Exception:
            first_txt = full_text or ""
        lines = [ln.strip() for ln in first_txt.splitlines() if ln.strip()]
        v = _kv_pick(lines, "address")
        if v:
            addr = _clean_address_like(v)
        elif name:
            # take next 3–5 lines after name, dropping metric/noise tokens
            for i, ln in enumerate(lines):
                if name.split()[0].lower() in ln.lower():
                    bucket = []
                    for lj in lines[i+1:i+6]:
                        if _LABEL_PAT.match(lj): break
                        cand = _clean_address_like(lj)
                        if cand:
                            bucket.append(cand)
                    if bucket:
                        addr = _clean_address_like(", ".join(bucket))
                    break

    if not branch:
        bb = _extract_first_match(r"(account\s*branch|branch\s*address)\s*[:\-]\s*(.+)", full_text, 2)
        branch = _clean_branch_addr(bb)

    account_holder = {"Name": name, "Address": addr, "ContactNumber": phone, "Email": email}
    bank_account   = {"AccountNumber": acct, "IFSC": ifsc, "BranchAddress": branch}

    meta_out = {"extracted_via": "agentic-doc"}
    if meta:
        try:
            meta_out["chunk_refs"] = {
                k: getattr(getattr(meta, k), "chunk_references", None)
                for k in ["name", "address", "contact_number", "email", "account_number", "ifsc", "branch_address"]
            }
        except Exception:
            pass
    return account_holder, bank_account, meta_out


# ---- Gemini fallback for details (prompt strengthened) ----
def call_gemini(prompt: str, model_name: str = None) -> str:
    if os.getenv("USE_GEMINI", "0") != "1" or genai is None:
        return "{}"
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "{}"
    model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    try:
        genai.configure(api_key=api_key)
        try:
            model = genai.GenerativeModel(
                model_name,
                generation_config={"response_mime_type": "application/json"}
            )
        except Exception:
            model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        return (getattr(resp, "text", None) or "{}").strip()
    except Exception as e:
        logging.warning(f"Gemini call failed: {e}")
        return "{}"

def surface_key_blobs(page_texts: List[str]) -> Dict[str, str]:
    header = page_texts[0] if page_texts else ""
    nearby = "\n\n".join(page_texts[1:3])
    return {"header": header, "nearby": nearby}

def extract_details_with_llm(blobs: Dict[str, str]) -> Tuple[Dict[str, Optional[str]], Dict[str, Optional[str]], Dict]:
    system_rules = (
        "You extract ONLY the following fields from bank-statement headers. "
        "Ignore totals/metrics like 'Cr Count', 'Debits', 'Credits', 'Closing Bal', 'Statement of Account', etc. "
        "Return STRICT JSON. If a field is missing, use null. Do not guess.\n"
        "Schema:\n"
        "{\n"
        '  "account_holder": {"Name": "...", "Address": "...", "ContactNumber": "...", "Email": "..."},\n'
        '  "bank_account": {"AccountNumber": "...", "IFSC": "...", "BranchAddress": "..."}\n'
        "}\n"
        "Constraints:\n"
        "- Name: Person/entity name only (no labels, no metrics).\n"
        "- Address: Postal address only. Remove tokens like 'Drawing Power', 'IFSC', 'Overdraft', 'Account Description', 'Closing Bal'.\n"
        "- ContactNumber: Indian 10-digit preferred.\n"
        "- IFSC: /^[A-Z]{4}0[A-Z0-9]{6}$/.\n"
        "- AccountNumber: 9–18 digits only.\n"
        "- BranchAddress: Branch/office address only."
    )
    user = f"HEADER:\n{blobs.get('header','')}\n\nNEARBY:\n{blobs.get('nearby','')}\n\nReturn JSON only per schema."
    out = _gemini_json_call(system_rules, user)
    try:
        data = json.loads(out)
    except Exception:
        return {}, {}, {"extracted_via": "gemini", "error": "invalid_json"}

    ah = data.get("account_holder", {}) if isinstance(data, dict) else {}
    ba = data.get("bank_account", {}) if isinstance(data, dict) else {}

    ah = {
        "Name": _clean_name(ah.get("Name")),
        "Address": _strip_noise_tokens(ah.get("Address"), "address"),
        "ContactNumber": _phone_prefer_contact(" ".join([blobs.get("header",""), blobs.get("nearby","")])) if not ah.get("ContactNumber") else _phone_prefer_contact(ah.get("ContactNumber")),
        "Email": _email_ok(ah.get("Email")),
    }
    ba = {
        "AccountNumber": _clean_acct(ba.get("AccountNumber")),
        "IFSC": _ifsc_ok(ba.get("IFSC")),
        "BranchAddress": _clean_branch_addr(ba.get("BranchAddress")),
    }
    return ah, ba, {"extracted_via": "gemini"}

# ---- Heuristic fallback for details ----
def extract_details_heuristic(page_texts: Iterable[str]) -> Tuple[Dict[str, Optional[str]], Dict[str, Optional[str]], Dict]:
    pages = list(page_texts)
    text_all = "\n".join(pages)
    first = pages[0] if pages else ""
    lines = [l.rstrip() for l in first.splitlines() if l.strip()]

    h_name   = _kv_pick(lines, "name")
    h_addr   = _kv_pick(lines, "address")
    h_phone  = _kv_pick(lines, "contact number") or _kv_pick(lines, "phone") or _kv_pick(lines, "mobile")
    h_email  = _kv_pick(lines, "email") or _kv_pick(lines, "e-mail")
    h_acct   = _kv_pick(lines, "account number") or _kv_pick(lines, "account no") or _kv_pick(lines, "a/c no")
    h_ifsc   = _kv_pick(lines, "ifsc")
    h_branch = _kv_pick(lines, "branch address") or _kv_pick(lines, "account branch") or _kv_pick(lines, "branch")

    name  = _clean_name(h_name) or _derive_name_from_email(h_email)
    addr  = _clean_address_like(h_addr)
    phone = _phone_prefer_contact(h_phone or "") or _phone_prefer_contact(text_all)
    email = _email_ok(h_email) or _email_ok(text_all)
    acct  = _clean_acct(h_acct) or _clean_acct(text_all)
    ifsc  = _ifsc_ok(h_ifsc) or _ifsc_ok(text_all)
    branch= _clean_branch_addr(h_branch)

    # If still no address, harvest 3–5 lines under a name-like line
    if (not addr or len(addr) < 8) and name:
        for i, ln in enumerate(lines):
            if name.split()[0].lower() in ln.lower():
                bucket = []
                for lj in lines[i+1:i+6]:
                    if _LABEL_PAT.match(lj): break
                    cand = _clean_address_like(lj)
                    if cand:
                        bucket.append(cand)
                if bucket:
                    addr = _clean_address_like(", ".join(bucket))
                break

    account_holder = {"Name": name, "Address": addr, "ContactNumber": phone, "Email": email}
    bank_account   = {"AccountNumber": acct, "IFSC": ifsc, "BranchAddress": branch}
    return account_holder, bank_account, {"extracted_via": "heuristic"}



# ======================= Stage 3b: Merge details =======================

def _pick_best_value(*candidates, kind: str = "") -> Optional[str]:
    for c in candidates:
        if not c:
            continue
        c0 = str(c).strip()
        if not c0:
            continue
        if kind == "ifsc" and not _ifsc_ok(c0):
            continue
        if kind == "acct" and not _clean_acct(c0):
            continue
        if kind == "email" and not _email_ok(c0):
            continue
        if kind == "phone":
            dig = re.sub(r"\D", "", c0)
            if len(dig) != 10 or dig[0] not in "6789":
                continue
            c0 = dig
        if kind == "address":
            c0 = _strip_noise_tokens(c0, "address")
        if kind == "branch":
            c0 = _clean_branch_addr(c0)
        if kind == "name":
            c0 = _clean_name(c0)
        if c0:
            return c0
    return None

def merge_details(
    agentic_ah: Dict[str, Optional[str]], agentic_ba: Dict[str, Optional[str]],
    gem_ah: Dict[str, Optional[str]], gem_ba: Dict[str, Optional[str]],
    heur_ah: Dict[str, Optional[str]], heur_ba: Dict[str, Optional[str]],
) -> Tuple[Dict[str, Optional[str]], Dict[str, Optional[str]], Dict[str, Any]]:
    ah = {
        "Name":          _pick_best_value(agentic_ah.get("Name"), heur_ah.get("Name"), gem_ah.get("Name"), kind="name"),
        "Address":       _pick_best_value(agentic_ah.get("Address"), heur_ah.get("Address"), gem_ah.get("Address"), kind="address"),
        "ContactNumber": _pick_best_value(agentic_ah.get("ContactNumber"), heur_ah.get("ContactNumber"), gem_ah.get("ContactNumber"), kind="phone"),
        "Email":         _pick_best_value(agentic_ah.get("Email"), heur_ah.get("Email"), gem_ah.get("Email"), kind="email"),
    }
    ba = {
        "AccountNumber": _pick_best_value(agentic_ba.get("AccountNumber"), heur_ba.get("AccountNumber"), gem_ba.get("AccountNumber"), kind="acct"),
        "IFSC":          _pick_best_value(agentic_ba.get("IFSC"), heur_ba.get("IFSC"), gem_ba.get("IFSC"), kind="ifsc"),
        "BranchAddress": _pick_best_value(agentic_ba.get("BranchAddress"), heur_ba.get("BranchAddress"), gem_ba.get("BranchAddress"), kind="branch"),
    }
    provenance = {
        "name_source":          ("agentic" if ah["Name"] == agentic_ah.get("Name") else "heuristic" if ah["Name"] == heur_ah.get("Name") else "gemini" if ah["Name"] == gem_ah.get("Name") else None),
        "address_source":       ("agentic" if ah["Address"] == agentic_ah.get("Address") else "heuristic" if ah["Address"] == heur_ah.get("Address") else "gemini" if ah["Address"] == gem_ah.get("Address") else None),
        "contact_source":       ("agentic" if ah["ContactNumber"] == agentic_ah.get("ContactNumber") else "heuristic" if ah["ContactNumber"] == heur_ah.get("ContactNumber") else "gemini" if ah["ContactNumber"] == gem_ah.get("ContactNumber") else None),
        "email_source":         ("agentic" if ah["Email"] == agentic_ah.get("Email") else "heuristic" if ah["Email"] == heur_ah.get("Email") else "gemini" if ah["Email"] == gem_ah.get("Email") else None),
        "acct_source":          ("agentic" if ba["AccountNumber"] == agentic_ba.get("AccountNumber") else "heuristic" if ba["AccountNumber"] == heur_ba.get("AccountNumber") else "gemini" if ba["AccountNumber"] == gem_ba.get("AccountNumber") else None),
        "ifsc_source":          ("agentic" if ba["IFSC"] == agentic_ba.get("IFSC") else "heuristic" if ba["IFSC"] == heur_ba.get("IFSC") else "gemini" if ba["IFSC"] == gem_ba.get("IFSC") else None),
        "branch_source":        ("agentic" if ba["BranchAddress"] == agentic_ba.get("BranchAddress") else "heuristic" if ba["BranchAddress"] == heur_ba.get("BranchAddress") else "gemini" if ba["BranchAddress"] == gem_ba.get("BranchAddress") else None),
    }
    return ah, ba, provenance


# ======================= Unify & Orchestrate =======================

def unify_results(
    account_holder: Dict[str, Optional[str]],
    bank_account: Dict[str, Optional[str]],
    transactions: pd.DataFrame,
    provenance: Dict[str, Any],
) -> Dict[str, Any]:
    tx = transactions.copy()
    col_order = [c for c in ["Date", "Description", "Ref", "Debit", "Credit", "Balance"] if c in tx.columns]
    tx = tx[col_order] if col_order else tx
    return {
        "account_holder": account_holder,
        "bank_account": bank_account,
        "transactions": tx.to_dict(orient="records"),
        "provenance": provenance,
    }

def process_pdf(path: str) -> Dict[str, Any]:
    validate_pdf(path)
    doc_profile = compute_doc_profile(path)
    page_texts = extract_page_texts(path)

    # ---- Stage 2: Transactions (Vision Agent → pdfplumber → HDFC → regex) ----
    transactions_df = pd.DataFrame()
    table_report: Dict[str, Any] = {}
    algorithmic_success = False
    llm_used_for_tables = False

    if os.getenv("USE_VISION_AGENT_TX", "1") == "1":
        try:
            tx_df_va, rep_va, ok_va = extract_transactions_with_agentic_doc(path)
            if ok_va and not tx_df_va.empty:
                transactions_df, table_report, algorithmic_success = tx_df_va, rep_va, True
                llm_used_for_tables = False
        except Exception as e:
            table_report = {"anomalies": [f"Vision Agent error: {e}"]}

    if transactions_df.empty:
        if _HAS_PDFPLUMBER:
            try:
                tx_df1, rep1, ok1 = _parse_transactions_pdfplumber(path)
                if ok1 and not tx_df1.empty:
                    transactions_df, table_report, algorithmic_success = tx_df1, rep1, True
            except Exception as e:
                table_report = {"anomalies": [f"pdfplumber error: {e}"]}

    if transactions_df.empty:
        tx_df_h, rep_h, ok_h = _parse_transactions_hdfc_like(page_texts)
        if ok_h and not tx_df_h.empty:
            transactions_df, table_report, algorithmic_success = tx_df_h, rep_h, True

    if transactions_df.empty:
        tx_df2, rep2, ok2 = _parse_transactions_regex(page_texts)
        if ok2 and not tx_df2.empty:
            transactions_df, table_report, algorithmic_success = tx_df2, rep2, True

    # ---- Stage 4: LLM fallback for tables (ONLY if algorithmic failed) ----
    # ---- Stage 4: LLM fallback for tables (ONLY if algorithmic failed) ----
    if transactions_df.empty and os.getenv("USE_LLM_TABLE_FALLBACK", "1") == "1":
        tx_llm, rep_llm, ok_llm = extract_transactions_with_llm(page_texts)
        if ok_llm and not tx_llm.empty:
            transactions_df, table_report, algorithmic_success = tx_llm, rep_llm, True
            llm_used_for_tables = True
    if algorithmic_success:
        logging.info(f"[TX] Parsed via: {'LLM' if llm_used_for_tables else 'Algorithmic'}; rows={len(transactions_df)}")
    else:
        logging.warning("[TX] No transactions parsed by any method")


    if not transactions_df.empty:
        transactions_df = transactions_df.loc[:, ~transactions_df.columns.duplicated()]

    # ---- Stage 3: Details (multi-source with merge) ----
    agentic_ah: Dict[str, Optional[str]] = {}
    agentic_ba: Dict[str, Optional[str]] = {}
    agentic_meta: Dict[str, Any] = {}

    gem_ah: Dict[str, Optional[str]] = {}
    gem_ba: Dict[str, Optional[str]] = {}
    gem_meta: Dict[str, Any] = {}

    heur_ah: Dict[str, Optional[str]] = {}
    heur_ba: Dict[str, Optional[str]] = {}
    heur_meta: Dict[str, Any] = {}

    # Agentic-Doc first
    use_agentic = bool(os.getenv("VISION_AGENT_API_KEY")) and os.getenv("USE_AGENTIC_DOC", "1") == "1"
    if use_agentic:
        try:
            agentic_ah, agentic_ba, agentic_meta = extract_details_with_agentic_doc(path)
        except Exception as e:
            agentic_meta = {"extracted_via": "agentic-doc", "error": str(e)}

    # Gemini on header blobs (always try; we merge field-wise)
    blobs = surface_key_blobs(page_texts)
    try:
        gem_ah, gem_ba, gem_meta = extract_details_with_llm(blobs)
    except Exception as e:
        gem_meta = {"extracted_via": "gemini", "error": str(e)}

    # Heuristic pass
    heur_ah, heur_ba, heur_meta = extract_details_heuristic(page_texts)

    # Merge fields
    account_json, bank_json, merge_src = merge_details(
        agentic_ah or {}, agentic_ba or {},
        gem_ah or {}, gem_ba or {},
        heur_ah or {}, heur_ba or {},
    )

    # ---------- SAFE FIELD VALIDATION ----------
    field_audit = {}
    try:
        if validate_all_fields_and_merge is not None:
            account_json = account_json or {}
            bank_json = bank_json or {}
            account_json, bank_json, field_audit = validate_all_fields_and_merge(
                page_texts, account_json, bank_json
            )
        else:
            field_audit = {"skipped": "validators_runner.py not found"}
    except Exception as e:
        field_audit = {"error": f"{type(e).__name__}: {e}"}

    # ---------- SAFE TRANSACTION VALIDATION ----------
    tx_audit = {}
    try:
        if validate_and_fix_transactions is not None and not transactions_df.empty:
            transactions_df, tx_audit = validate_and_fix_transactions(transactions_df)
        else:
            tx_audit = {"skipped": "transactions_validator.py not found or no transactions"}
    except Exception as e:
        tx_audit = {"error": f"{type(e).__name__}: {e}"}

    provenance = {
        "algorithmic_success": algorithmic_success,
        "llm_used_for_tables": llm_used_for_tables,
        "coverage": table_report.get("coverage_hits", 0),
        "anomalies": table_report.get("anomalies", []),
        "pages_spanned": table_report.get("pages_spanned", []),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "field_validation": field_audit,
        "doc_profile": doc_profile,
        "llm_metadata": {
            "agentic": agentic_meta,
            "gemini": gem_meta,
            "heuristic": heur_meta,
            "merge_sources": merge_src
        },
        "transactions_validation": tx_audit,
    }

    return unify_results(account_json, bank_json, transactions_df, provenance)
# ---- API adapter (keep at the very bottom of pipeline.py) ----
def run_pipeline(file_bytes: bytes, use_gemini: bool = True, llm_fallback: bool = True) -> dict:
    """
    API-facing canonical entrypoint expected by api.py.

    - Accepts PDF bytes from FastAPI
    - Writes to a temp file
    - Calls your existing process_pdf(pdf_path)
    - Returns the unified dict
    """
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        result = process_pdf(tmp_path)
        if not isinstance(result, dict):
            raise RuntimeError("process_pdf() did not return a dict")
        return result
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
# ---------- Public entrypoints for server.py ----------
def extract_from_pdf(file_bytes: bytes, **kwargs) -> dict:
    """
    Main entrypoint for API server. Accepts raw PDF bytes and returns structured result.
    Ignores extra kwargs (llm_fallback, use_gemini, etc.).
    """
    import tempfile, os
    fd, tmp = tempfile.mkstemp(suffix=".pdf")
    try:
        with open(tmp, "wb") as f:
            f.write(file_bytes)
        # ⚠️ Call process_pdf WITHOUT passing kwargs
        return process_pdf(tmp)
    finally:
        try: os.close(fd)
        except: pass
        try: os.unlink(tmp)
        except: pass


def process_pdf_bytes(file_bytes: bytes, **kwargs) -> dict:
    """Alias for extract_from_pdf for compatibility."""
    return extract_from_pdf(file_bytes)


def extract_from_path(path: str, **kwargs) -> dict:
    """For direct file path usage."""
    return process_pdf(path)
