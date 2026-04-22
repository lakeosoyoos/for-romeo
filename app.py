"""
For Romeo April 21 — Unidirectional Shortened Duplicate Report with Loss.

Upload unidirectional SOR files (one direction). The app auto-detects the
common filename prefix + fiber numbers, compares every pair, and renders the
shortened report (same format and pagination as the latest
ZachRequestShortenedReportWithLoss, adapted for a single direction).

Run:  streamlit run app.py
"""
import os
import re
import sys
import tempfile
import shutil
from collections import Counter

import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from report_core import (
    load_fiber, compare_pairs, build_report,
    html_to_pdf_bytes, TOP_N,
)


st.set_page_config(
    page_title="For Romeo April 21 — Unidirectional Shortened Report",
    layout="wide",
)

st.title("Unidirectional Shortened Duplicate Report (with Loss)")
st.caption(
    "Upload a set of SOR files from a single direction. "
    "The app auto-detects the filename prefix and fiber numbers, then builds "
    "the shortened report (top "
    f"{TOP_N} pairs per ranking, multi-page tables with repeating headers)."
)

# ----- sidebar -----------------------------------------------------------
with st.sidebar:
    st.header("Report settings")
    route_name = st.text_input("Route name (used in report title)",
                               value="Route")
    direction_label = st.text_input("Direction label (e.g. DNWRCH→RCHDNW, A→B)",
                                    value="A→B")
    st.divider()
    st.subheader("Filename parsing")
    prefix_mode = st.radio(
        "Prefix detection",
        ["Auto-detect", "Manual"],
        index=0,
        help="Auto-detect finds the common prefix before the fiber number.",
    )
    manual_prefix = st.text_input(
        "Manual prefix (if Manual selected)",
        value="",
        help="Everything before the fiber number. Example: 'DNWRCH-A-'",
    )
    manual_suffix = st.text_input(
        "Suffix before .sor (optional)",
        value="",
        help="Example: '_1550' for files like LSC1LSC60001_1550.sor",
    )

# ----- upload -----------------------------------------------------------
uploads = st.file_uploader(
    "Drop .sor files here (one direction, any count)",
    type=["sor"],
    accept_multiple_files=True,
)

if not uploads:
    st.info("Waiting for SOR files…")
    st.stop()


# ----- stash uploads to a temp dir --------------------------------------
tmp_dir = tempfile.mkdtemp(prefix="romeo_sor_")
saved_paths = []
for uf in uploads:
    dst = os.path.join(tmp_dir, uf.name)
    with open(dst, "wb") as fh:
        fh.write(uf.getbuffer())
    saved_paths.append(dst)

st.success(f"Received {len(saved_paths)} SOR file(s).")


# ----- figure out prefix/suffix/fiber numbers ---------------------------
def detect_prefix_suffix(filenames):
    """Return (prefix, suffix, {fiber_num: filename}) by finding the digit
    run inside each filename and taking the most common (prefix, suffix)
    pair across the set."""
    pat = re.compile(r"^(?P<pre>.*?)(?P<num>\d+)(?P<suf>[^/]*)\.sor$", re.IGNORECASE)
    combos = Counter()
    parsed = []
    for fn in filenames:
        base = os.path.basename(fn)
        # prefer the LAST digit run to survive prefixes that already contain digits
        m = None
        for candidate in re.finditer(r"\d+", base[: -len(".sor")] if base.lower().endswith(".sor") else base):
            m = candidate
        if not m or not base.lower().endswith(".sor"):
            continue
        pre = base[: m.start()]
        suf = base[m.end(): -len(".sor")]
        num = int(m.group(0))
        combos[(pre, suf)] += 1
        parsed.append((pre, suf, num, base))
    if not combos:
        return None, None, {}
    (best_pre, best_suf), _ = combos.most_common(1)[0]
    fiber_map = {}
    for pre, suf, num, base in parsed:
        if pre == best_pre and suf == best_suf:
            fiber_map[num] = base
    return best_pre, best_suf, fiber_map


if prefix_mode == "Auto-detect":
    prefix, suffix, fiber_map = detect_prefix_suffix([os.path.basename(p) for p in saved_paths])
    if not fiber_map:
        st.error("Could not auto-detect a fiber-number pattern. Switch to Manual.")
        st.stop()
else:
    prefix = manual_prefix
    suffix = manual_suffix
    pat = re.compile(
        r"^" + re.escape(prefix) + r"(\d+)" + re.escape(suffix) + r"\.sor$",
        re.IGNORECASE,
    )
    fiber_map = {}
    for p in saved_paths:
        base = os.path.basename(p)
        m = pat.match(base)
        if m:
            fiber_map[int(m.group(1))] = base
    if not fiber_map:
        st.error(f"No files matched prefix '{prefix}' suffix '{suffix}'.")
        st.stop()

fiber_nums = sorted(fiber_map.keys())
with st.expander(f"Parsed {len(fiber_nums)} fibers  ·  prefix = '{prefix}'  ·  suffix = '{suffix}'", expanded=False):
    for n in fiber_nums:
        st.text(f"  fiber {n:>5}  →  {fiber_map[n]}")

if len(fiber_nums) < 2:
    st.warning("Need at least 2 fibers to form pairs.")
    st.stop()


# ----- parse & compare ---------------------------------------------------
with st.spinner("Parsing SOR files…"):
    fibers = {}
    for n in fiber_nums:
        fp = os.path.join(tmp_dir, fiber_map[n])
        try:
            fibers[f"{n:04d}"] = load_fiber(fp)
        except Exception as e:
            st.warning(f"Failed to parse {fiber_map[n]}: {e}")

if len(fibers) < 2:
    st.error("Not enough fibers parsed to run comparisons.")
    st.stop()

with st.spinner(f"Comparing {len(fibers)*(len(fibers)-1)//2} pairs…"):
    pairs = compare_pairs(fibers)

n_pairs = len(pairs)
median_diff = sorted(p['max_diff_mdB'] for p in pairs)[n_pairs // 2] if pairs else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Fibers", len(fibers))
c2.metric("Pairs", n_pairs)
c3.metric("Closest pair (mdB)", f"{pairs[0]['max_diff_mdB']:.0f}" if pairs else "—")
c4.metric("Median max diff (mdB)", f"{median_diff:.0f}" if pairs else "—")


# ----- build report ------------------------------------------------------
html = build_report(fibers, pairs, route_name, direction_label, fiber_nums)

safe_route = re.sub(r"[^A-Za-z0-9]+", "_", route_name).strip("_") or "report"
html_path = os.path.join(tmp_dir, f"{safe_route}_shortened.html")
with open(html_path, "w", encoding="utf-8") as fh:
    fh.write(html)

st.subheader("Preview")
st.components.v1.html(html, height=900, scrolling=True)

st.subheader("Download")
dl1, dl2 = st.columns(2)
with dl1:
    st.download_button(
        "Download HTML",
        data=html.encode("utf-8"),
        file_name=f"{safe_route}_shortened.html",
        mime="text/html",
    )

with dl2:
    if "pdf_bytes" not in st.session_state or st.session_state.get("pdf_html") != html:
        if st.button("Generate PDF"):
            with st.spinner("Rendering PDF…"):
                try:
                    pdf_bytes = html_to_pdf_bytes(html, base_url=tmp_dir)
                    st.session_state["pdf_bytes"] = pdf_bytes
                    st.session_state["pdf_html"] = html
                    st.rerun()
                except Exception as e:
                    st.error(f"PDF export failed: {e}")
    if st.session_state.get("pdf_bytes") is not None and st.session_state.get("pdf_html") == html:
        st.download_button(
            "Download PDF",
            data=st.session_state["pdf_bytes"],
            file_name=f"{safe_route}_shortened.pdf",
            mime="application/pdf",
        )

st.caption(f"Working directory: {tmp_dir}")
