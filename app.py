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
    load_fiber, compare_pairs, build_combined_report,
    html_to_pdf_bytes, TOP_N,
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
    """Handle names like `DNWRCH-A-271.sor` and `LSC1LSC60001_1550.sor`
    (wavelength suffix) alike. For each file, enumerate every digit run in
    the stem as a candidate fiber-number position. Then rank (prefix, suffix)
    tuples by how many files share them — the best-scoring tuple for each
    file is the one we use. A wavelength suffix like `_1550` shows up in
    every file's best tuple, so it's correctly treated as part of the
    suffix rather than the fiber number."""
    per_file_options = []  # list of list[(pre, suf, num, base)]
    for fn in filenames:
        base = os.path.basename(fn)
        if not base.lower().endswith(".sor"):
            continue
        stem = base[:-4]
        options = []
        for m in re.finditer(r"\d+", stem):
            pre = stem[: m.start()]
            suf = stem[m.end():]
            num = int(m.group(0))
            options.append((pre, suf, num, base))
        if options:
            per_file_options.append(options)

    # Score each (prefix, suffix) by how many files mention it at all.
    tuple_counts = Counter()
    for options in per_file_options:
        # Each file may list the same (pre, suf) only once naturally, but
        # dedupe just in case.
        for pre, suf, _, _ in set((o[0], o[1]) for o in options):
            tuple_counts[(pre, suf)] += 1

    # For each file, assign it to the highest-scoring tuple available for it.
    groups = defaultdict(dict)
    for options in per_file_options:
        best = max(options, key=lambda o: tuple_counts[(o[0], o[1])])
        pre, suf, num, base = best
        # Only accept assignments where the shared-tuple count is >= 2
        # (avoids attaching a stray single-file interpretation to a group).
        if tuple_counts[(pre, suf)] < 2:
            continue
        groups[(pre, suf)][num] = base
    return groups


groups = group_by_direction([os.path.basename(p) for p in saved_paths])
groups = {k: v for k, v in groups.items() if len(v) >= 2}

if not groups:
    st.error(
        "Could not find any prefix with ≥2 fibers. Check that the uploaded "
        "files share a consistent naming pattern."
    )
    with st.expander("Sample filenames (for debugging)", expanded=False):
        for p in saved_paths[:25]:
            st.text(os.path.basename(p))
    st.stop()


# ----- parse each direction ---------------------------------------------
def pick_common(values):
    vals = [v for v in values if v]
    if not vals:
        return ""
    return Counter(vals).most_common(1)[0][0]


def process_direction(prefix, suffix, fiber_map):
    fiber_nums = sorted(fiber_map.keys())
    fibers = {}
    for n in fiber_nums:
        fp = os.path.join(tmp_dir, fiber_map[n])
        try:
            fibers[f"{n:04d}"] = load_fiber(fp)
        except Exception as e:
            st.warning(f"Failed to parse {fiber_map[n]}: {e}")
    if len(fibers) < 2:
        return None

    gp_list = [fibers[k].get('gen_params', {}) or {} for k in fibers]
    loc_a = pick_common([g.get('location_a', '') for g in gp_list])
    loc_b = pick_common([g.get('location_b', '') for g in gp_list])
    cable_id = pick_common([g.get('cable_id', '') for g in gp_list])
    cable_code = pick_common([g.get('cable_code', '') for g in gp_list])
    clean_prefix = prefix.rstrip("-_ ") or "Route"
    label = f"{loc_a} → {loc_b}" if loc_a and loc_b else clean_prefix
    route_hint = cable_id or cable_code or clean_prefix

    pairs = compare_pairs(fibers)
    return {
        'label': label,
        'pairs': pairs,
        'fiber_nums': fiber_nums,
        'route_hint': route_hint,
        'loc_a': loc_a, 'loc_b': loc_b,
        'prefix': prefix,
    }


directions = []
for (pre, suf), fiber_map in groups.items():
    with st.spinner(f"Parsing {len(fiber_map)} fibers from '{pre or '∅'}'…"):
        d = process_direction(pre, suf, fiber_map)
    if d is not None:
        directions.append(d)

if not directions:
    st.error("No direction had at least 2 parseable fibers.")
    st.stop()

# If two directions ended up with the same GenParams label (firmware quirk
# where A/B locations don't match the filename convention), fall back to
# the cleaned filename prefix so sections are distinguishable.
label_counts = Counter(d['label'] for d in directions)
collisions = {label for label, c in label_counts.items() if c > 1}
for d in directions:
    if d['label'] in collisions:
        clean = d['prefix'].rstrip('-_ ') or 'Direction'
        if d['loc_a'] and d['loc_b']:
            d['label'] = f"{clean} ({d['loc_a']} → {d['loc_b']})"
        else:
            d['label'] = clean


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

st.download_button(
    "Download combined PDF",
    data=st.session_state["pdf_bytes"],
    file_name=fname,
    mime="application/pdf",
)
