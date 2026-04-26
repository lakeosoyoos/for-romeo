"""
For Romeo April 21 — Unidirectional Shortened Duplicate Report with Loss.

Upload either a .zip of SOR files or the .sor files directly. Every distinct
filename prefix is treated as its own direction; a single combined PDF is
produced with one section per direction (each section has its own banner,
histogram, and three ranking tables).

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
    load_fiber_records, compare_pairs, build_combined_report,
    html_to_pdf_bytes, debug_timestamp_candidates, TOP_N,
)


st.set_page_config(
    page_title="For Romeo April 21 — Shortened Duplicate Report",
    layout="wide",
)

# ----- password gate ----------------------------------------------------
APP_PASSWORD = st.secrets.get("app_password", "") if hasattr(st, "secrets") else ""

if not st.session_state.get("authed"):
    st.title("Shortened Duplicate Report")
    if not APP_PASSWORD:
        st.error(
            "App password is not configured. Set `app_password` in Streamlit "
            "Cloud → app **Settings** → **Secrets**, or in a local "
            "`.streamlit/secrets.toml` for development."
        )
        st.stop()
    pwd = st.text_input("Password", type="password")
    if pwd == APP_PASSWORD:
        st.session_state["authed"] = True
        st.rerun()
    elif pwd:
        st.error("Incorrect password.")
    st.stop()

st.title("Shortened Duplicate Report")
st.caption(
    "Upload a .zip of SOR files or the .sor files directly. The app detects "
    "every distinct filename prefix, treats each one as a direction, and "
    f"builds a single combined PDF with one section per direction. Top {TOP_N} "
    "pairs per ranking, multi-page tables with repeating headers."
)

# ----- upload -----------------------------------------------------------
# Nonce is appended to the uploader's key so pressing "Clear" forces a fresh
# widget, which drops the currently-uploaded files from Streamlit's state.
if "uploader_nonce" not in st.session_state:
    st.session_state["uploader_nonce"] = 0

uploads = st.file_uploader(
    "Drop .sor, .json, .trc files and/or a .zip here",
    type=["sor", "json", "trc", "zip"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state['uploader_nonce']}",
)

# Clear button sits immediately under the uploader so it's obvious.
if st.button("🗑  Clear uploaded files", type="secondary"):
    for k in ("pdf_bytes", "pdf_html"):
        st.session_state.pop(k, None)
    st.session_state["uploader_nonce"] += 1
    st.rerun()

if not uploads:
    st.info("Waiting for files…")
    st.stop()


# ----- stash uploads to a temp dir --------------------------------------
tmp_dir = tempfile.mkdtemp(prefix="romeo_sor_")


SUPPORTED_EXTS = (".sor", ".json", ".trc")


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
                if not inner.lower().endswith(SUPPORTED_EXTS):
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
    st.error("No .sor / .json / .trc files found in the uploads.")
    st.stop()

# Partition by extension.
sor_paths = [p for p in saved_paths if p.lower().endswith(".sor")]
json_paths = [p for p in saved_paths if p.lower().endswith(".json")]
trc_paths = [p for p in saved_paths if p.lower().endswith(".trc")]

summary = f"Loaded {len(saved_paths)} file(s): {len(sor_paths)} SOR"
if json_paths:
    summary += f", {len(json_paths)} JSON"
if trc_paths:
    summary += f", {len(trc_paths)} TRC"
st.success(summary + ".")

processed_paths = sor_paths + json_paths + trc_paths
if not processed_paths:
    st.error("No SOR / JSON / TRC files found; nothing to report.")
    st.stop()

saved_paths = processed_paths


# ----- group files by firmware GenParams (fallback to filename) --------
def _filename_prefix_key(path, fiber_num=None):
    """Return the filename prefix preceding the fiber number. If `fiber_num`
    is given, we find the digit run that matches it (so a trailing wavelength
    suffix like `_1550` can't be mistaken for the fiber number). Otherwise we
    use the FIRST digit run in the stem, not the last."""
    base = os.path.basename(path)
    # Strip extension (.sor / .json / .trc)
    stem, _, ext = base.rpartition('.')
    if not stem:
        stem = base
    matches = list(re.finditer(r'\d+', stem))
    if not matches:
        return stem.rstrip('-_ ') or 'Route'
    chosen = None
    if fiber_num is not None:
        for m in matches:
            if int(m.group(0)) == fiber_num:
                chosen = m
                break
    if chosen is None:
        chosen = matches[0]
    return stem[: chosen.start()].rstrip('-_ ') or 'Route'


def parse_and_group(paths):
    """Single-pass parse + group. For each file, call load_fiber_records
    (which emits one record per wavelength — TRC fans out to N records,
    SOR/JSON emit a single record). Bucket each record into
        groups[(loc_a, loc_b, wavelength)] = {fiber_num: fiber_record}
    using GenParams locations when both are present, else the filename
    prefix.

    Returns (groups, skipped).
    """
    groups = defaultdict(dict)
    skipped = []
    for p in paths:
        try:
            records = load_fiber_records(p)
        except Exception as e:
            skipped.append((p, {'error': str(e)}))
            continue
        if not records:
            skipped.append((p, {'error': 'no records returned'}))
            continue
        for wl_nm, rec in records:
            gp = rec.get('gen_params', {}) or {}
            loc_a = (gp.get('location_a') or '').strip()
            loc_b = (gp.get('location_b') or '').strip()
            fiber_id_raw = (gp.get('fiber_id') or '').strip()
            wl = wl_nm or gp.get('wavelength_code', 0) or 0

            digits_from_id = re.sub(r'\D', '', fiber_id_raw)
            fnum = int(digits_from_id) if digits_from_id else None
            if fnum is None:
                m = re.search(r'\d+', os.path.basename(p))
                fnum = int(m.group(0)) if m else None
            if fnum is None:
                skipped.append((p, gp))
                continue

            if loc_a and loc_b:
                key = (loc_a, loc_b, wl)
            else:
                prefix = _filename_prefix_key(p, fiber_num=fnum)
                key = (prefix, '<fromfilename>', wl)

            groups[key][fnum] = rec
    return groups, skipped


with st.spinner(f"Parsing {len(saved_paths)} file(s)…"):
    fw_groups, skipped = parse_and_group(saved_paths)

fw_groups = {k: v for k, v in fw_groups.items() if len(v) >= 2}

if not fw_groups:
    st.error(
        "Could not find any direction with ≥2 fibers. "
        f"{len(skipped)} file(s) had unusable metadata."
    )
    with st.expander("Skip reasons (for debugging)", expanded=False):
        for p, info in skipped[:25]:
            st.text(f"{os.path.basename(p)}  →  {info}")
    st.stop()

if skipped:
    st.warning(f"Skipped {len(skipped)} file(s) (missing metadata or parse error).")


# ----- parse each direction ---------------------------------------------
def pick_common(values):
    vals = [v for v in values if v]
    if not vals:
        return ""
    return Counter(vals).most_common(1)[0][0]


FALLBACK_SENTINEL = '<fromfilename>'


def process_direction(loc_a, loc_b, wavelength, fiber_records):
    """fiber_records: {fiber_num: parsed_fiber_dict}. Builds the direction
    record from the already-parsed records (no I/O here)."""
    fiber_nums = sorted(fiber_records.keys())
    fibers = {f"{n:04d}": fiber_records[n] for n in fiber_nums}
    if len(fibers) < 2:
        return None

    gp_list = [fibers[k].get('gen_params', {}) or {} for k in fibers]
    cable_id = pick_common([g.get('cable_id', '') for g in gp_list])
    cable_code = pick_common([g.get('cable_code', '') for g in gp_list])

    if loc_b == FALLBACK_SENTINEL:
        label_base = loc_a or 'Unnamed'
        loc_a_display, loc_b_display = label_base, ''
    else:
        label_base = f"{loc_a} → {loc_b}"
        loc_a_display, loc_b_display = loc_a, loc_b

    label = f"{label_base} @ {wavelength}nm" if wavelength else label_base
    route_hint = cable_id or cable_code or loc_a_display or 'Route'

    pairs = compare_pairs(fibers)
    return {
        'label': label,
        'pairs': pairs,
        'fiber_nums': fiber_nums,
        'route_hint': route_hint,
        'loc_a': loc_a_display, 'loc_b': loc_b_display,
        'wavelength': wavelength,
    }


# If any JSON files were uploaded, surface a debug view of their timestamp
# keys so we can confirm they're being parsed. Collapsed by default.
if json_paths:
    first_json = json_paths[0]
    with st.expander(f"Debug: timestamp keys in first JSON file "
                     f"({os.path.basename(first_json)})", expanded=False):
        for path, value in debug_timestamp_candidates(first_json)[:40]:
            st.text(f"  {path}  =  {value!r}")

directions = []
for (loc_a, loc_b, wl), fiber_records in fw_groups.items():
    d = process_direction(loc_a, loc_b, wl, fiber_records)
    if d is not None:
        directions.append(d)

if not directions:
    st.error("No direction had at least 2 parseable fibers.")
    st.stop()

# Sort directions for consistent output: by wavelength then A→B alphabetical.
directions.sort(key=lambda d: (d['wavelength'], d['loc_a'], d['loc_b']))


# ----- derive a single route name across all directions ----------------
route_name = pick_common([d['route_hint'] for d in directions]) or "Route"


# ----- summary ----------------------------------------------------------
st.write(
    f"Detected **{len(directions)} direction(s)** in route **{route_name}**."
)

cols = st.columns(len(directions))
for col, d in zip(cols, directions):
    pairs = d['pairs']
    closest = f"{pairs[0]['max_diff_mdB']:.0f}" if pairs else "—"
    median_val = (sorted(p['max_diff_mdB'] for p in pairs)[len(pairs) // 2]
                  if pairs else 0)
    col.markdown(f"**{d['label']}**")
    col.metric("Fibers", len(d['fiber_nums']))
    col.metric("Pairs", len(pairs))
    col.metric("Closest (mdB)", closest)
    col.metric("Median (mdB)", f"{median_val:.0f}")


# ----- build combined report + PDF --------------------------------------
html = build_combined_report(route_name, [{
    'label': d['label'],
    'pairs': d['pairs'],
    'fiber_nums': d['fiber_nums'],
} for d in directions])

if st.session_state.get("pdf_html") != html:
    with st.spinner("Rendering combined PDF…"):
        try:
            st.session_state["pdf_bytes"] = html_to_pdf_bytes(html, base_url=tmp_dir)
            st.session_state["pdf_html"] = html
        except Exception as e:
            st.error(f"PDF export failed: {e}")
            st.stop()

safe_route = re.sub(r"[^A-Za-z0-9]+", "_", route_name).strip("_") or "route"
fname = f"{safe_route}_bidir_shortened.pdf"

# Style the download button to look like Streamlit's green success alert:
# darker green semi-transparent background, lime-green text, full-width bar.
st.markdown(
    """
    <style>
    [data-testid="stDownloadButton"] > button {
        background-color: rgba(33, 195, 84, 0.16) !important;
        color: rgb(33, 195, 84) !important;
        border: 1px solid rgba(33, 195, 84, 0.30) !important;
        border-radius: 0.5rem !important;
        padding: 0.75rem 1rem !important;
        width: 100% !important;
        font-weight: 500 !important;
        transition: background-color 0.15s ease, border-color 0.15s ease;
    }
    [data-testid="stDownloadButton"] > button:hover {
        background-color: rgba(33, 195, 84, 0.26) !important;
        border-color: rgba(33, 195, 84, 0.55) !important;
        color: rgb(33, 195, 84) !important;
    }
    [data-testid="stDownloadButton"] > button:focus,
    [data-testid="stDownloadButton"] > button:active {
        background-color: rgba(33, 195, 84, 0.30) !important;
        color: rgb(33, 195, 84) !important;
        box-shadow: none !important;
        outline: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.download_button(
    "Download combined PDF",
    data=st.session_state["pdf_bytes"],
    file_name=fname,
    mime="application/pdf",
)
