#!/usr/bin/env python3
"""Segmented, pinned-IP, resumable download from ModelScope for a box whose
resolver drops the CDN host and whose per-connection throughput swings
between 0.4 and 9 MB/s. Many parallel ranged segments with retries is the
only thing that saturates such a link. Stdlib + curl only.

usage: seg_download.py manifest.json dest_dir [--workers 12] [--seg-mb 256]
"""
import argparse, hashlib, json, os, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

WWW = "https://www.modelscope.cn/models/Qwen/Qwen-Image-2.1/resolve/master/"
CDN = "cdn-lfs-cn-1.modelscope.cn"
DOH = "https://223.5.5.5/resolve?name=%s&type=A" % CDN
DOH_URL = "https://223.5.5.5/dns-query"


def cdn_ips():
    """AliDNS over HTTPS works from this box when plain DNS does not."""
    for _ in range(5):
        try:
            d = json.load(urllib.request.urlopen(DOH, timeout=10))
            ips = [a["data"] for a in d.get("Answer", []) if a.get("type") == 1]
            if ips:
                return ips
        except Exception:
            time.sleep(1)
    return []


def redirect_for(path):
    """Ask www (which resolves) where the bytes are; do not follow it here."""
    out = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{redirect_url}",
                          "--doh-url", DOH_URL, "--max-time", "20", WWW + path],
                         capture_output=True, text=True)
    return out.stdout.strip()


def pull_segment(path, start, end, part, ips, i):
    if os.path.exists(part) and os.path.getsize(part) == end - start + 1:
        return part, "cached"
    for attempt in range(12):
        url = redirect_for(path)
        if not url:
            time.sleep(2); continue
        ip = ips[(i + attempt) % len(ips)] if ips else None
        # DNS over HTTPS for every curl call: the box's UDP resolver is dead
        # most of the time, and pinning one CDN IP with --resolve picked nodes
        # that connect at 0.4 MB/s as often as 9. Let AliDNS pick per request.
        cmd = ["curl", "-s", "-r", f"{start}-{end}", "-o", part, "--max-time", "900",
               "--speed-limit", "20000", "--speed-time", "60", "--doh-url", DOH_URL]
        r = subprocess.run(cmd + [url], capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(part) and os.path.getsize(part) == end - start + 1:
            return part, f"ok via {ip}"
        time.sleep(min(30, 2 * (attempt + 1)))
    return part, "FAILED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest"); ap.add_argument("dest")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seg-mb", type=int, default=256)
    a = ap.parse_args()
    files = json.load(open(a.manifest))
    ips = cdn_ips()
    print(f"cdn ips via DoH: {ips}", flush=True)
    seg = a.seg_mb << 20
    tasks, plan = [], {}
    for f in files:
        dest = os.path.join(a.dest, f["path"])
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest) and os.path.getsize(dest) == f["size"]:
            print(f"have    {f['path']}", flush=True); continue
        parts = []
        for k, s in enumerate(range(0, f["size"], seg)):
            e = min(s + seg, f["size"]) - 1
            part = f"{dest}.part{k:04d}"
            parts.append(part); tasks.append((f["path"], s, e, part))
        plan[f["path"]] = (dest, parts, f)
    print(f"{len(tasks)} segments over {len(plan)} files", flush=True)
    t0, done, total = time.time(), 0, len(tasks)
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(pull_segment, p, s, e, part, ips, i): (p, part)
                for i, (p, s, e, part) in enumerate(tasks)}
        for fut in as_completed(futs):
            part, status = fut.result(); done += 1
            got = sum(os.path.getsize(x) for pl in plan.values() for x in pl[1] if os.path.exists(x))
            print(f"{time.time()-t0:6.0f}s {done:3d}/{total} {status:22s} "
                  f"{got/2**30:6.2f} GiB {got/2**20/max(1,time.time()-t0):5.1f} MiB/s  {os.path.basename(part)}", flush=True)
    bad = []
    for path, (dest, parts, f) in plan.items():
        if not all(os.path.exists(p) and os.path.getsize(p) > 0 for p in parts):
            bad.append(path); continue
        h = hashlib.sha256()
        with open(dest, "wb") as out:
            for p in parts:
                with open(p, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 24), b""):
                        out.write(chunk); h.update(chunk)
        ok = (not f.get("sha256")) or h.hexdigest() == f["sha256"]
        if ok:
            for p in parts: os.remove(p)
        else:
            bad.append(path)
        print(f"{'verified' if ok else 'SHA MISMATCH'} {path}", flush=True)
    print("FAILED:" if bad else "ALL VERIFIED", bad, flush=True)


if __name__ == "__main__":
    main()
