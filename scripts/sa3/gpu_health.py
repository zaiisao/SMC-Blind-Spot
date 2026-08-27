"""Report which GPUs are safe to use, by PCIe link health.

Why this exists: on 2026-08-23 the machine hard-locked at 11:15 during a 4-GPU
generation run. GPU 1 (0000:2e:00.0) had trained its PCIe link down to x2 of 16
and was the only device accumulating AER correctable errors -- 8 RxErr plus one
NonFatalErr within 24 minutes of an idle boot. Sustained host<->device traffic
on a link that marginal escalates corrected errors to fatal and takes the box
down. Exclude such GPUs rather than rediscovering this the hard way.

Usage:  eval "$(python3 scripts/sa3/gpu_health.py --export)"   # sets CUDA_VISIBLE_DEVICES
        python3 scripts/sa3/gpu_health.py                      # human-readable table
"""
import argparse, subprocess, sys
from pathlib import Path

# A link that has degraded below this fraction of its advertised width is
# treated as faulty regardless of error counts.
MIN_WIDTH_FRAC = 0.5


def probe():
    q = ("index,pci.bus_id,pcie.link.width.current,pcie.link.width.max")
    out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    gpus = []
    for line in out.strip().splitlines():
        idx, bus, cur, mx = [f.strip() for f in line.split(",")]
        # nvidia-smi prints 00000000:2E:00.0; sysfs wants 0000:2e:00.0
        dev = bus[4:].lower()
        cor = Path(f"/sys/bus/pci/devices/{dev}/aer_dev_correctable")
        nonfatal = Path(f"/sys/bus/pci/devices/{dev}/aer_dev_nonfatal")
        errs = 0
        for f in (cor, nonfatal):
            if f.exists():
                for l in f.read_text().splitlines():
                    k, _, v = l.partition(" ")
                    if k == "TOTAL_ERR_COR" or k.endswith("ERR_FATAL"):
                        continue
                    errs += int(v or 0)
        cur, mx = int(cur), int(mx)
        healthy = errs == 0 and cur >= MIN_WIDTH_FRAC * mx
        gpus.append(dict(index=int(idx), dev=dev, width=cur, width_max=mx,
                         aer=errs, healthy=healthy))
    return gpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true",
                    help="emit a CUDA_VISIBLE_DEVICES export line and nothing else")
    a = ap.parse_args()
    gpus = probe()
    ok = [g for g in gpus if g["healthy"]]
    if a.export:
        if not ok:
            print("echo 'no healthy GPUs' >&2; false")
            return 1
        print("export CUDA_VISIBLE_DEVICES=" + ",".join(str(g["index"]) for g in ok))
        return 0
    for g in gpus:
        print(f"gpu {g['index']}  {g['dev']}  x{g['width']}/{g['width_max']}  "
              f"aer={g['aer']}  {'OK' if g['healthy'] else 'EXCLUDED'}")
    print(f"\nhealthy: {[g['index'] for g in ok]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
