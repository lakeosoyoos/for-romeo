"""
lsa_event_calculator.py
=======================
Two-Point LSA event calculator that matches EXFO's desktop software.

Reads EXFO SOR files and recalculates splice loss and reflectance using
Least Squares Approximation on the raw trace data, using the fitting
window boundaries stored in the KeyEvents block.

USAGE
-----
    python3 lsa_event_calculator.py <file.sor>
    python3 lsa_event_calculator.py <file.sor> --verbose
    python3 lsa_event_calculator.py <folder/> --csv results.csv

REQUIREMENTS
------------
    pip install numpy
"""

import os
import sys
import struct
import zlib
import argparse
import csv

import numpy as np

C_LIGHT = 2.998e8   # m/s
SP      = 50e-9     # EXFO sampling period = 50 ns


# ═══════════════════════════════════════════════════════════════════════════════
#  SOR PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_blocks(data):
    """Parse the SOR Map block directory."""
    off = 0
    try:
        name_end = data.index(b'\x00', off) + 1
        off = name_end + 6
        num_blocks = struct.unpack_from('<H', data, off)[0]
        off += 2
        block_list = []
        for _ in range(num_blocks):
            ne = data.index(b'\x00', off) + 1
            nm = data[off:ne - 1].decode('latin-1')
            bv = struct.unpack_from('<H', data, ne)[0]
            bs = struct.unpack_from('<I', data, ne + 2)[0]
            block_list.append((nm, bv, bs))
            off = ne + 6

        seen = set()
        blocks = {}
        search_from = name_end + 2 + 4
        for nm, bv, bs in block_list:
            if nm in seen:
                continue
            seen.add(nm)
            needle = nm.encode('latin-1') + b'\x00'
            idx = data.find(needle, search_from)
            if idx >= 0:
                blocks[nm] = {
                    'offset': idx, 'size': bs, 'ver': bv,
                    'body': idx + len(needle),
                }
                search_from = idx + bs
    except Exception:
        blocks = {}
    return blocks


def _read_ior(data):
    """Scan for IOR value."""
    for off in range(0, min(len(data), 1000)):
        try:
            val = struct.unpack_from('<I', data, off)[0]
            if 145000 <= val <= 149000:
                return val / 100000.0
        except struct.error:
            pass
    return 1.46820


def _read_fxd_params(data, blocks):
    """Parse FxdParams for wavelength and acquisition range."""
    if 'FxdParams' not in blocks:
        return {}
    body = blocks['FxdParams']['body']
    date_time = struct.unpack_from('<I', data, body)[0]
    units = data[body + 4:body + 6].decode('latin-1')
    wavelength = struct.unpack_from('<H', data, body + 6)[0]  # 0.1 nm
    num_pw = struct.unpack_from('<H', data, body + 16)[0]
    pw_end = body + 18 + num_pw * 2
    acq_range = struct.unpack_from('<I', data, pw_end)[0]  # 100 ps units
    return {
        'date_time': date_time,
        'wavelength_nm': wavelength / 10.0,
        'acq_range_100ps': acq_range,
    }


def _parse_key_events(data, blocks):
    """Parse KeyEvents with full fitting window boundaries."""
    if 'KeyEvents' not in blocks:
        return []
    body = blocks['KeyEvents']['body']
    num_events = struct.unpack_from('<H', data, body)[0]
    pos = body + 2

    events = []
    for _ in range(num_events):
        evnum      = struct.unpack_from('<H', data, pos)[0];  pos += 2
        tot        = struct.unpack_from('<I', data, pos)[0];  pos += 4
        slope      = struct.unpack_from('<h', data, pos)[0];  pos += 2
        splice     = struct.unpack_from('<h', data, pos)[0];  pos += 2
        refl       = struct.unpack_from('<i', data, pos)[0];  pos += 4
        evt_raw    = data[pos:pos + 8];                       pos += 8
        end_prev   = struct.unpack_from('<I', data, pos)[0];  pos += 4
        start_curr = struct.unpack_from('<I', data, pos)[0];  pos += 4
        end_curr   = struct.unpack_from('<I', data, pos)[0];  pos += 4
        start_next = struct.unpack_from('<I', data, pos)[0];  pos += 4
        peak_curr  = struct.unpack_from('<I', data, pos)[0];  pos += 4
        pos += 2  # padding

        evt_type = evt_raw.split(b'\x00')[0].decode('latin-1', errors='replace')
        is_reflective = evt_type[:1] == '1'
        is_end = evt_type[1:2] == 'E'

        events.append({
            'number':         evnum,
            'time_of_travel': tot,
            'slope_raw':      slope,
            'splice_loss_fw': splice / 1000.0,
            'reflection_fw':  refl / 1000.0,
            'type':           evt_type,
            'is_reflective':  is_reflective,
            'is_end':         is_end,
            'end_prev':       end_prev,
            'start_curr':     start_curr,
            'end_curr':       end_curr,
            'start_next':     start_next,
            'peak':           peak_curr,
        })
    return events


def _read_rawsamples(data, IOR, blocks):
    """
    Read EXFO RawSamples from ExfoNewProprietaryBlock.
    Returns (trace_signal, dx_m, i_conn) or None.
    trace_signal is in signal convention (higher = stronger, millidB).
    """
    blk_name = None
    for name in blocks:
        if 'ExfoNewProprietaryBlock' in name:
            blk_name = name
            break
    if blk_name is None:
        return None

    exfo = blocks[blk_name]
    body = exfo['body']
    end = exfo['offset'] + exfo['size']
    chunk = data[body:end]

    # Decompress the proprietary block
    s0 = None
    for skip in (40, 36):
        try:
            s0 = zlib.decompress(chunk[skip:])
            break
        except zlib.error:
            pass
    if s0 is None:
        return None

    # Find RawSamples inside decompressed data
    rs_off = s0.find(b'RawSamples')
    if rs_off < 0:
        return None

    avail = len(s0) - (rs_off + 15)
    n = avail // 2
    if n < 100:
        return None

    raw = (np.frombuffer(s0[rs_off + 15: rs_off + 15 + n * 2], dtype='<u2')
           .astype(np.float64) / 1000.0)  # millidB → dB

    dx_m = SP * C_LIGHT / (2 * IOR)

    # Find launch connector Fresnel peak for reference
    i_conn = int(np.argmax(raw[:400]))

    # Return FULL untrimmed trace — LSA needs samples before launch connector
    # for fitting windows. i_conn tells us where distance 0 is.
    return raw, dx_m, i_conn


def _read_datapts(data, blocks, IOR):
    """
    Read DataPts block as fallback (higher density, ~2.5 m/sample).
    Returns (trace_array, dx_m, n_points) or None.
    Values in arbitrary power units (NOT dB).
    """
    if 'DataPts' not in blocks:
        return None

    body = blocks['DataPts']['body']
    num_pts = struct.unpack_from('<I', data, body)[0]
    scale = struct.unpack_from('<H', data, body + 4)[0]
    if scale == 0:
        scale = 1000

    pts_start = body + 6
    if len(data) < pts_start + num_pts * 2:
        return None

    raw = np.frombuffer(data, dtype='<u2', count=num_pts, offset=pts_start)
    trace = raw.astype(np.float64) / scale

    # DataPts dx
    fxd_pos = data.find(b'FxdParams\x00', 100)
    if fxd_pos >= 0:
        off = fxd_pos + len('FxdParams\x00')
        off += 4 + 2 + 2 + 8  # date, units, wl, offsets
        num_pw = struct.unpack_from('<H', data, off)[0]; off += 2
        off += 2  # first pulse width
        dx_raw = struct.unpack_from('<I', data, off)[0]
        dx_m = dx_raw * 1e-10 * C_LIGHT / (2 * IOR)
    else:
        dx_m = SP * C_LIGHT / (2 * IOR)

    return trace, dx_m, num_pts


# ═══════════════════════════════════════════════════════════════════════════════
#  LSA CORE
# ═══════════════════════════════════════════════════════════════════════════════

def time_to_dist_m(time_100ps, IOR):
    """Convert 100-ps propagation time to distance in meters.
    EXFO stores one-way propagation time in KeyEvents."""
    return time_100ps * 1e-10 * C_LIGHT / IOR


def time_to_sample_idx(time_100ps):
    """Convert 100-ps propagation time to a RawSamples index.
    RawSamples spacing = SP = 50 ns. Time units = 100 ps.
    index = time_100ps * 100ps / SP = time_100ps / 500."""
    return time_100ps / 500.0


def dist_to_index(dist_m, dx_m):
    """Convert distance in meters to a float sample index."""
    return dist_m / dx_m


def fit_segment(trace, idx_start, idx_end):
    """
    Fit a least-squares line to trace[idx_start:idx_end].
    Returns (slope, intercept, n_points) or None if < 2 points.
    """
    i0 = max(0, int(round(idx_start)))
    i1 = min(len(trace), int(round(idx_end)))
    if i1 <= i0:
        return None
    segment = trace[i0:i1]
    n = len(segment)
    if n < 2:
        return None
    x = np.arange(i0, i0 + n, dtype=np.float64)
    coeffs = np.polyfit(x, segment, 1)
    return coeffs[0], coeffs[1], n


def lsa_splice_loss(trace, dx_m, IOR, event, events, event_idx, i_conn=0):
    """
    Compute Two-Point LSA splice loss for one event.

    Uses the fitting window boundaries from the KeyEvents block:
      BEFORE: trace[end_prev → start_curr]
      AFTER:  trace[end_curr → start_next]

    i_conn: the launch connector offset that was trimmed from the raw trace.
            Event times are absolute, so we subtract this offset.

    Returns dict with LSA results.
    """
    def time_to_idx(t):
        """Convert 100-ps propagation time to sample index in untrimmed trace.
        Event time=0 corresponds to the launch connector at raw index i_conn."""
        if t > 0xFFF00000:
            return -1.0
        return time_to_sample_idx(t) + i_conn

    ep = time_to_idx(event['end_prev'])
    sc = time_to_idx(event['start_curr'])
    ec = time_to_idx(event['end_curr'])
    sn = time_to_idx(event['start_next'])
    pk = time_to_idx(event['peak'])
    evt_idx = time_to_idx(event['time_of_travel'])

    # Fit BEFORE segment
    before_fit = fit_segment(trace, ep, sc)

    # Fit AFTER segment
    after_fit = fit_segment(trace, ec, sn)

    # Calculate splice loss
    # In signal convention: BEFORE line > AFTER line means positive loss
    # splice_loss = before_at_event - after_at_event
    splice_loss = None
    if event_idx == 0 and before_fit and before_fit[2] >= 10:
        # First event with good BEFORE data (non-trimmed file with launch lead).
        # Use standard Two-Point LSA but extend the AFTER if it's too short.
        bm, bb, bn = before_fit
        before_at_event = bm * evt_idx + bb

        if after_fit and after_fit[2] >= 5:
            # Enough AFTER points for direct fit
            am, ab, an = after_fit
            after_at_event = am * evt_idx + ab
        elif event_idx + 1 < len(events):
            # Short AFTER segment: extend using the next event's AFTER (long backscatter)
            next_evt = events[event_idx + 1]
            next_ec = time_to_idx(next_evt['end_curr'])
            next_sn = time_to_idx(next_evt['start_next'])
            long_after = fit_segment(trace, next_ec, next_sn)
            if after_fit and long_after:
                # Weighted average of short and long extrapolations
                am_s, ab_s, an_s = after_fit
                am_l, ab_l, an_l = long_after
                short_at = am_s * evt_idx + ab_s
                long_at = am_l * evt_idx + ab_l
                # Weight by point count
                after_at_event = (short_at * an_s + long_at * an_l) / (an_s + an_l)
            elif long_after:
                am_l, ab_l, _ = long_after
                after_at_event = am_l * evt_idx + ab_l
            else:
                am, ab, an = after_fit
                after_at_event = am * evt_idx + ab
        else:
            am, ab, an = after_fit
            after_at_event = am * evt_idx + ab

        splice_loss = before_at_event - after_at_event
    elif event_idx == 0:
        # First event, no good BEFORE (trimmed file).
        # Cannot compute accurate Event #1 loss without incoming backscatter.
        # Best effort: report as None (use firmware value).
        splice_loss = None
    elif before_fit and after_fit:
        bm, bb, bn = before_fit
        am, ab, an = after_fit
        before_at_event = bm * evt_idx + bb
        after_at_event = am * evt_idx + ab
        splice_loss = before_at_event - after_at_event
    elif before_fit and not after_fit:
        # Last event or missing AFTER: report from BEFORE only
        bm, bb, bn = before_fit
        before_at_event = bm * evt_idx + bb
        actual = trace[max(0, min(int(round(evt_idx)), len(trace)-1))]
        splice_loss = before_at_event - actual

    # Calculate reflectance (peak above backscatter level)
    reflectance = None
    peak_sample = int(round(pk))
    if 0 <= peak_sample < len(trace) and before_fit:
        bm, bb, bn = before_fit
        backscatter_at_peak = bm * pk + bb
        # In signal convention: peak is higher than backscatter
        # In loss convention: peak is LOWER (less loss = more signal)
        reflectance = trace[peak_sample] - backscatter_at_peak

    # Attenuation (slope of AFTER segment in dB/km)
    attenuation = None
    if after_fit:
        am, ab, an = after_fit
        attenuation = am * (1000.0 / dx_m)  # dB per km

    return {
        'splice_loss_lsa': splice_loss,
        'reflectance_lsa': reflectance,
        'attenuation_lsa': attenuation,
        'before_npts': before_fit[2] if before_fit else 0,
        'after_npts': after_fit[2] if after_fit else 0,
        'before_slope': before_fit[0] * (1000.0 / dx_m) if before_fit else None,
        'after_slope': after_fit[0] * (1000.0 / dx_m) if after_fit else None,
        'evt_sample_idx': evt_idx,
        'ep': ep, 'sc': sc, 'ec': ec, 'sn': sn, 'pk': pk,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  HIGH-LEVEL API
# ═══════════════════════════════════════════════════════════════════════════════

def parse_sor_with_windows(filepath):
    """
    Parse a SOR file and extract everything needed for LSA.
    Returns dict with trace, events, metadata.
    """
    with open(filepath, 'rb') as f:
        data = f.read()

    blocks = _parse_blocks(data)
    IOR = _read_ior(data)
    fxd = _read_fxd_params(data, blocks)
    events = _parse_key_events(data, blocks)

    # Try RawSamples first
    rs = _read_rawsamples(data, IOR, blocks)
    trace = None
    dx_m = None
    trace_source = None
    i_conn = 0
    if rs:
        trace_signal, dx_m, i_conn = rs
        # trace_signal is full untrimmed in signal convention (higher = stronger)
        # Keep in signal convention for LSA — fits work the same either way,
        # and we avoid confusion about sign conventions.
        trace = trace_signal.astype(np.float64)
        trace_source = 'RawSamples'

    # Fallback to DataPts
    if trace is None:
        dp = _read_datapts(data, blocks, IOR)
        if dp:
            trace_raw, dx_m, n_pts = dp
            trace = (trace_raw[0] - trace_raw).astype(np.float64)
            trace_source = 'DataPts'

    return {
        'filepath': filepath,
        'IOR': IOR,
        'dx_m': dx_m,
        'wavelength_nm': fxd.get('wavelength_nm', 0),
        'events': events,
        'trace': trace,
        'trace_source': trace_source,
        'n_points': len(trace) if trace is not None else 0,
        'i_conn': i_conn,  # launch connector sample offset in raw data
    }


def calculate_all_events(filepath, verbose=False):
    """
    Parse a SOR file and compute LSA splice loss for all events.
    Returns list of event result dicts.
    """
    parsed = parse_sor_with_windows(filepath)
    if parsed['trace'] is None:
        return []

    trace = parsed['trace']
    dx_m = parsed['dx_m']
    IOR = parsed['IOR']
    events = parsed['events']
    i_conn = parsed.get('i_conn', 0)

    # Use first event as distance reference (like EXFO does)
    first_dist_m = time_to_dist_m(events[0]['time_of_travel'], IOR) if events else 0

    results = []
    for i, evt in enumerate(events):
        dist_m = time_to_dist_m(evt['time_of_travel'], IOR)
        dist_km = (dist_m - first_dist_m) / 1000.0

        if i > 0:
            prev_dist = time_to_dist_m(events[i-1]['time_of_travel'], IOR)
            span_km = (dist_m - prev_dist) / 1000.0
        else:
            span_km = 0.0

        lsa = lsa_splice_loss(trace, dx_m, IOR, evt, events, i, i_conn)

        result = {
            'number': evt['number'],
            'type': evt['type'],
            'is_reflective': evt['is_reflective'],
            'is_end': evt['is_end'],
            'dist_km': dist_km,
            'span_km': span_km,
            # Firmware values (from KeyEvents)
            'splice_fw': evt['splice_loss_fw'],
            'refl_fw': evt['reflection_fw'],
            'slope_fw': evt['slope_raw'] / 1000.0,
            # LSA values
            'splice_lsa': lsa['splice_loss_lsa'],
            'refl_lsa': lsa['reflectance_lsa'],
            'atten_lsa': lsa['attenuation_lsa'],
            # Fitting quality
            'before_npts': lsa['before_npts'],
            'after_npts': lsa['after_npts'],
            'before_slope': lsa['before_slope'],
            'after_slope': lsa['after_slope'],
        }

        if verbose:
            print(f"  Event #{evt['number']} at {dist_km:.4f} km:")
            print(f"    Windows: ep={lsa['ep']:.1f} sc={lsa['sc']:.1f} "
                  f"ec={lsa['ec']:.1f} sn={lsa['sn']:.1f}")
            print(f"    Before: {lsa['before_npts']} pts, "
                  f"After: {lsa['after_npts']} pts")
            if lsa['splice_loss_lsa'] is not None:
                print(f"    LSA splice: {lsa['splice_loss_lsa']:+.3f} dB  "
                      f"(firmware: {evt['splice_loss_fw']:+.3f} dB)")

        results.append(result)

    return results


def format_event_table(results, filename=''):
    """Format results as an EXFO-style event table."""
    lines = []
    if filename:
        lines.append(f"\n  {filename}")
    lines.append(f"  {'─' * 90}")
    lines.append(
        f"  {'#':>3}  {'Type':<10}  {'Dist (km)':>10}  {'Span (km)':>10}  "
        f"{'Loss LSA':>10}  {'Loss FW':>10}  {'Refl':>10}  {'Atten':>10}")
    lines.append(f"  {'─' * 90}")

    for r in results:
        lsa_str = f"{r['splice_lsa']:+.3f}" if r['splice_lsa'] is not None else '---'
        fw_str = f"{r['splice_fw']:+.3f}"
        refl_str = f"{r['refl_lsa']:.3f}" if r['refl_lsa'] is not None else f"{r['refl_fw']:.3f}"
        atten_str = f"{r['atten_lsa']:.3f}" if r['atten_lsa'] is not None else '---'
        lines.append(
            f"  {r['number']:>3}  {r['type']:<10}  {r['dist_km']:>10.4f}  "
            f"{r['span_km']:>10.4f}  {lsa_str:>10}  {fw_str:>10}  "
            f"{refl_str:>10}  {atten_str:>10}")

    lines.append(f"  {'─' * 90}")
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def collect_sor_files(inputs):
    """Expand directories to .sor files."""
    out = []
    for inp in inputs:
        if os.path.isdir(inp):
            for fn in sorted(os.listdir(inp)):
                if fn.lower().endswith('.sor'):
                    out.append(os.path.join(inp, fn))
        elif os.path.isfile(inp):
            out.append(inp)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Two-Point LSA event calculator for EXFO SOR files.")
    parser.add_argument('inputs', nargs='+',
                        help='SOR files or directories')
    parser.add_argument('--csv', metavar='PATH',
                        help='Export results to CSV')
    parser.add_argument('--verbose', action='store_true',
                        help='Show fitting details')
    args = parser.parse_args()

    filepaths = collect_sor_files(args.inputs)
    if not filepaths:
        print("No SOR files found.")
        sys.exit(1)

    print(f"Processing {len(filepaths)} SOR file(s)...")

    all_results = []
    for fp in filepaths:
        basename = os.path.basename(fp)
        results = calculate_all_events(fp, verbose=args.verbose)
        if results:
            print(format_event_table(results, basename))
            all_results.append((basename, results))

    if args.csv and all_results:
        fieldnames = [
            'filename', 'event', 'type', 'dist_km', 'span_km',
            'splice_lsa', 'splice_fw', 'refl_lsa', 'refl_fw',
            'atten_lsa', 'before_npts', 'after_npts',
        ]
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for fname, results in all_results:
                for r in results:
                    w.writerow({
                        'filename': fname,
                        'event': r['number'],
                        'type': r['type'],
                        'dist_km': round(r['dist_km'], 4),
                        'span_km': round(r['span_km'], 4),
                        'splice_lsa': round(r['splice_lsa'], 3) if r['splice_lsa'] is not None else '',
                        'splice_fw': round(r['splice_fw'], 3),
                        'refl_lsa': round(r['refl_lsa'], 3) if r['refl_lsa'] is not None else '',
                        'refl_fw': round(r['refl_fw'], 3),
                        'atten_lsa': round(r['atten_lsa'], 3) if r['atten_lsa'] is not None else '',
                        'before_npts': r['before_npts'],
                        'after_npts': r['after_npts'],
                    })
        print(f"\nCSV saved to: {args.csv}")


if __name__ == '__main__':
    main()
