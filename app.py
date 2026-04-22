"""
For Romeo April 21 — Unidirectional Shortened Duplicate Report with Loss.

Upload either a .zip of unidirectional SOR files or the .sor files directly.
The app auto-detects the filename prefix, fiber numbers, and reads the
route/direction labels from the SOR GenParams block. No manual settings.

Run:  streamlit run app.py
"""
import os
import re
import sys
import tempfile
import zipfile
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
    "Upload a .zip of SOR files or the .sor files directly. The app reads the "
    "route name, direction, and fiber numbers from the files themselves — no "
    "settings to configure. Top "
    f"{TOP_N} pairs per ranking, multi-page tables with repeating headers."
)

# ----- upload -----------------------------------------------------------
uploads = st.file_uploader(
    "Drop .sor files and/or a .zip here",
    type=["sor", "zip"],
    accept_multiple_files=True,
)

if not uploads:
    st.info("Waiting for files…")
    st.stop()


# ----- stash uploads to a temp dir --------------------------------------
tmp_dir = tempfile.mkdtemp(prefix="romeo_sor_")


def _extract(uf, dest_dir):
    """Save one upload to disk. Zip files are expanded (flat — .sor files
    only). Returns a list of saved .sor file paths."""
    name = uf.name
    data = uf.getbuffer()
    if name.lower().endswith(".zip"):
        zpath = os.path.join(dest_dir, name)
        with open(zpath, "wb") as fh:
            fh.write(data)
        saved = []
        with zipfile.ZipFile(zpath) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                inner = os.path.basename(info.filename)
                if not inner.lower().endswith(".sor"):
                    continue
                dst = os.path.join(dest_dir, inner)
                with zf.open(info) as src, open(dst, "wb") as out:
                    out.write(src.read())
                saved.append(dst)
        os.remove(zpath)
        return saved
    else:
        dst = os.path.join(dest_dir, name)
        with open(dst, "wb") as fh:
            fh.write(data)
        return [dst]


saved_paths = []
for uf in uploads:
    saved_paths.extend(_extract(uf, tmp_dir))

if not saved_paths:
    st.error("No .sor files found in the uploads.")
    st.stop()

st.success(f"Loaded {len(saved_paths)} SOR file(s).")


# ----- auto-detect prefix/suffix/fiber numbers --------------------------
def detect_prefix_suffix(filenames):
    pat_stem = re.compile(r"\.sor$", re.IGNORECASE)
    combos = Counter()
    parsed = []
    for fn in filenames:
        base = os.path.basename(fn)
        if not pat_stem.search(base):
            continue
        stem = base[:-4]  # drop '.sor'
        m = None
        for candidate in re.finditer(r"\d+", stem):
            m = candidate  # last digit run
        if not m:
            continue
        pre = stem[: m.start()]
        suf = stem[m.end():]
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


prefix, suffix, fiber_map = detect_prefix_suffix(
    [os.path.basename(p) for p in saved_paths]
)
if not fiber_map or len(fiber_map) < 2:
    st.error(
        "Could not detect a consistent fiber-number pattern, or fewer than 2 "
        "fibers matched. Check that the uploaded files share a common naming."
    )
    st.stop()

fiber_nums = sorted(fiber_map.keys())


# ----- parse & compare ---------------------------------------------------
with st.spinner(f"Parsing {len(fiber_nums)} SOR files…"):
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


# ----- derive route name and direction label from GenParams --------------
def pick_common(values):
    """Return the most common non-empty value, or '' if none."""
    vals = [v for v in values if v]
    if not vals:
        return ""
    return Counter(vals).most_common(1)[0][0]


gp_list = [fibers[k].get('gen_params', {}) or {} for k in fibers]
cable_id = pick_common([g.get('cable_id', '') for g in gp_list])
loc_a = pick_common([g.get('location_a', '') for g in gp_list])
loc_b = pick_common([g.get('location_b', '') for g in gp_list])
cable_code = pick_common([g.get('cable_code', '') for g in gp_list])

route_name = cable_id or cable_code or (prefix.rstrip("-_ ") or "Route")
if loc_a and loc_b:
    direction_label = f"{loc_a} → {loc_b}"
else:
    direction_label = (prefix.rstrip("-_ ") or "Direction")


# ----- summary ----------------------------------------------------------
with st.spinner(f"Comparing {len(fibers)*(len(fibers)-1)//2} pairs…"):
    pairs = compare_pairs(fibers)

n_pairs = len(pairs)
median_diff = sorted(p['max_diff_mdB'] for p in pairs)[n_pairs // 2] if pairs else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Route", route_name[:18] + ("…" if len(route_name) > 18 else ""))
c2.metric("Direction", direction_label[:22] + ("…" if len(direction_label) > 22 else ""))
c3.metric("Fibers", len(fibers))
c4.metric("Pairs", n_pairs)

c5, c6, c7 = st.columns(3)
c5.metric("Closest pair (mdB)", f"{pairs[0]['max_diff_mdB']:.0f}" if pairs else "—")
c6.metric("Median max diff (mdB)", f"{median_diff:.0f}" if pairs else "—")
c7.metric("Prefix detected", prefix or "—")

with st.expander("Fibers parsed (click for detail)", expanded=False):
    for n in fiber_nums:
        gp = fibers.get(f"{n:04d}", {}).get('gen_params', {}) or {}
        cid = gp.get('cable_id', '')
        fid = gp.get('fiber_id', '')
        st.text(f"  fiber {n:>5}  →  {fiber_map[n]}   "
                f"[cable={cid!s} fiber_id={fid!s}]")


# ----- build report ------------------------------------------------------
html = build_report(fibers, pairs, route_name, direction_label, fiber_nums)

safe_route = re.sub(r"[^A-Za-z0-9]+", "_", route_name).strip("_") or "report"

# Cache the PDF so repeated downloads don't re-render.
if st.session_state.get("pdf_html") != html:
    with st.spinner("Rendering PDF…"):
        try:
            st.session_state["pdf_bytes"] = html_to_pdf_bytes(html, base_url=tmp_dir)
            st.session_state["pdf_html"] = html
        except Exception as e:
            st.error(f"PDF export failed: {e}")
            st.stop()

st.download_button(
    "Download PDF",
    data=st.session_state["pdf_bytes"],
    file_name=f"{safe_route}_shortened.pdf",
    mime="application/pdf",
)
