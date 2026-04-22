"""
For Romeo April 21 — Unidirectional Shortened Duplicate Report with Loss.

Upload either a .zip of SOR files or the .sor files directly. The app:
  1. Finds every distinct filename prefix (each = one direction).
  2. For every direction with >=2 fibers, builds a separate report
     and renders a PDF.
  3. Shows one download button per direction.

Run:  streamlit run app.py
"""
import os
import re
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict

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
    "Upload a .zip of SOR files or the .sor files directly. Every distinct "
    "filename prefix is treated as its own direction; one PDF is produced "
    f"per direction. Top {TOP_N} pairs per ranking, multi-page tables with "
    "repeating headers."
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


# ----- group files by (prefix, suffix) = direction ----------------------
def group_by_direction(filenames):
    """Return {(prefix, suffix): {fiber_num: basename}} for every distinct
    prefix/suffix pair found in the uploaded files."""
    groups = defaultdict(dict)
    for fn in filenames:
        base = os.path.basename(fn)
        if not base.lower().endswith(".sor"):
            continue
        stem = base[:-4]
        m = None
        for candidate in re.finditer(r"\d+", stem):
            m = candidate  # last digit run → fiber number
        if not m:
            continue
        pre = stem[: m.start()]
        suf = stem[m.end():]
        num = int(m.group(0))
        groups[(pre, suf)][num] = base
    return groups


groups = group_by_direction([os.path.basename(p) for p in saved_paths])
# Drop groups with fewer than 2 fibers (no pairs possible)
groups = {k: v for k, v in groups.items() if len(v) >= 2}

if not groups:
    st.error(
        "Could not find any prefix with ≥2 fibers. Check that the uploaded "
        "files share a consistent naming pattern."
    )
    st.stop()

st.write(
    f"Detected **{len(groups)} direction(s)**: "
    + ", ".join(
        f"`{(pre or '∅')}…{suf}` ({len(fm)} fibers)"
        for (pre, suf), fm in groups.items()
    )
)


# ----- process each direction -------------------------------------------
def pick_common(values):
    vals = [v for v in values if v]
    if not vals:
        return ""
    return Counter(vals).most_common(1)[0][0]


def build_direction(prefix, suffix, fiber_map):
    """Parse + compare + build HTML for one direction. Returns (label_dict,
    html_str) or (None, None) if something failed."""
    fiber_nums = sorted(fiber_map.keys())
    fibers = {}
    for n in fiber_nums:
        fp = os.path.join(tmp_dir, fiber_map[n])
        try:
            fibers[f"{n:04d}"] = load_fiber(fp)
        except Exception as e:
            st.warning(f"Failed to parse {fiber_map[n]}: {e}")
    if len(fibers) < 2:
        return None, None

    gp_list = [fibers[k].get('gen_params', {}) or {} for k in fibers]
    loc_a = pick_common([g.get('location_a', '') for g in gp_list])
    loc_b = pick_common([g.get('location_b', '') for g in gp_list])
    cable_id = pick_common([g.get('cable_id', '') for g in gp_list])
    cable_code = pick_common([g.get('cable_code', '') for g in gp_list])

    clean_prefix = prefix.rstrip("-_ ") or "Route"
    route_name = cable_id or cable_code or clean_prefix
    direction_label = f"{loc_a} → {loc_b}" if loc_a and loc_b else clean_prefix

    pairs = compare_pairs(fibers)
    html = build_report(fibers, pairs, route_name, direction_label, fiber_nums)

    median_diff = (sorted(p['max_diff_mdB'] for p in pairs)[len(pairs) // 2]
                   if pairs else 0)
    info = {
        'prefix': prefix,
        'suffix': suffix,
        'route_name': route_name,
        'direction_label': direction_label,
        'n_fibers': len(fibers),
        'n_pairs': len(pairs),
        'closest_mdB': pairs[0]['max_diff_mdB'] if pairs else 0,
        'median_mdB': median_diff,
        'loc_a': loc_a, 'loc_b': loc_b,
    }
    return info, html


# Cache PDFs in session state keyed by (prefix, suffix)
if "pdf_cache" not in st.session_state:
    st.session_state["pdf_cache"] = {}


for (pre, suf), fiber_map in groups.items():
    with st.spinner(f"Processing {pre or '∅'} ({len(fiber_map)} fibers)…"):
        info, html = build_direction(pre, suf, fiber_map)

    if info is None:
        st.warning(f"Skipped direction '{pre}' — not enough parseable fibers.")
        continue

    st.divider()
    st.subheader(f"{info['route_name']} — {info['direction_label']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fibers", info['n_fibers'])
    c2.metric("Pairs", info['n_pairs'])
    c3.metric("Closest (mdB)", f"{info['closest_mdB']:.0f}")
    c4.metric("Median (mdB)", f"{info['median_mdB']:.0f}")

    cache_key = (pre, suf)
    cache = st.session_state["pdf_cache"]
    if cache.get(cache_key, {}).get("html") != html:
        try:
            pdf_bytes = html_to_pdf_bytes(html, base_url=tmp_dir)
            cache[cache_key] = {"html": html, "pdf": pdf_bytes}
        except Exception as e:
            st.error(f"PDF export failed for {pre}: {e}")
            continue

    safe_route = re.sub(r"[^A-Za-z0-9]+", "_", info['route_name']).strip("_") or "route"
    dir_slug = (
        f"{info['loc_a']}_to_{info['loc_b']}" if info['loc_a'] and info['loc_b']
        else (pre.rstrip('-_ ') or 'dir')
    )
    dir_slug = re.sub(r"[^A-Za-z0-9]+", "_", dir_slug).strip("_") or "dir"
    fname = f"{safe_route}_{dir_slug}_shortened.pdf"

    st.download_button(
        f"Download PDF — {info['direction_label']}",
        data=cache[cache_key]["pdf"],
        file_name=fname,
        mime="application/pdf",
        key=f"dl_{pre}_{suf}",
    )
