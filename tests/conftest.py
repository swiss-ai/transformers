import pytest
import sys
import importlib

def _get_xielu_metrics():
    import os
    target = os.path.join('tests', 'activations', 'test_xielu_parity.py')
    for module in sys.modules.values():
        file = getattr(module, '__file__', '')
        if file and file.endswith(target):
            return getattr(module, 'XIELU_METRICS', {})
    return {}

@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print XIELU metrics in the pytest short summary"""
    metrics = _get_xielu_metrics()
    if not metrics:
        print("DEBUG: no metrics")
        return

    terminalreporter.ensure_newline()
    terminalreporter.write_sep("=", "XIELU Parity Summary")
    terminalreporter.write_line(f"Prompt: {metrics['Prompt']}")
    terminalreporter.write_line(f"Text ✗ VL: {metrics['Text (no vector loads)']}")
    terminalreporter.write_line(f"Text ✓ VL: {metrics['Text (with vector loads)']}")
    terminalreporter.write_line(f"ROUGE-L: {metrics['ROUGE-L']}")

    # Latencies
    lat_nv_20 = metrics['lat_nv_20']
    lat_vl_20 = metrics['lat_vl_20']
    # 1k latencies may be None if --1k not passed
    lat_nv_1k = metrics.get('lat_nv_1k')
    lat_vl_1k = metrics.get('lat_vl_1k')
    # Compute speedup for 20-token
    su20 = lat_nv_20 / lat_vl_20
    # Compute speedup for 1k-token if data present
    su1k = (lat_nv_1k / lat_vl_1k) if (lat_vl_1k is not None and lat_nv_1k is not None) else None

    GREEN = '\x1b[32m'; RED = '\x1b[31m'; RESET = '\x1b[0m'
    def color_latency(vl, nv):
        if vl < nv:
            return f"{GREEN}{vl:.2f} ms{RESET}", f"{RED}{nv:.2f} ms{RESET}"
        else:
            return f"{RED}{vl:.2f} ms{RESET}", f"{GREEN}{nv:.2f} ms{RESET}"
    cv20, cn20 = color_latency(lat_vl_20, lat_nv_20)
    if lat_vl_1k is not None and lat_nv_1k is not None:
        cv1k, cn1k = color_latency(lat_vl_1k, lat_nv_1k)
    else:
        cv1k, cn1k = "", ""

    def color_speedup(su):
        if su is None:
            return ""
        return f"{GREEN if su>1 else RED}{su:.2f}×{RESET}"
    csu20 = color_speedup(su20)
    csu1k = color_speedup(su1k)

    # Build table dynamically with right-justified columns, stripping ANSI codes for width measurement
    import re
    ansi_re = re.compile(r'\x1b\[[0-9;]*m')
    headers = ["", "20 tk", "1k tk"]
    rows = [
        ["✗ VL", cn20, cn1k],
        ["✓ VL", cv20, cv1k],
        ["", csu20, csu1k],
    ]
    # Compute max visible width for each column
    col_values = list(zip(headers, *rows))
    widths = [
        max(len(ansi_re.sub('', str(v))) for v in col)
        for col in col_values
    ]
    # Header row
    header_line = "| " + " | ".join(h.rjust(w) for h, w in zip(headers, widths)) + " |"
    sep_line = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    terminalreporter.write_line(header_line)
    terminalreporter.write_line(sep_line)
    # Data rows
    for row in rows:
        cells = []
        for cell, w in zip(row, widths):
            text = str(cell)
            clean = ansi_re.sub('', text)
            pad = w - len(clean)
            cells.append(' ' * pad + text)
        line = "| " + " | ".join(cells) + " |"
        terminalreporter.write_line(line)
