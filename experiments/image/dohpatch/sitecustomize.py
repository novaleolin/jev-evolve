"""Resolve hostnames over HTTPS when the box's UDP DNS is dead.

On the shared training node, the system resolvers (223.5.5.5, 114.114.114.114
over UDP/53) fail most of the time while HTTPS to 223.5.5.5 works. Editing
/etc/resolv.conf needs root. This patches socket.getaddrinfo in-process to ask
AliDNS's JSON API by IP, so every Python tool (pip, hf, modelscope) works
unchanged: PYTHONPATH=<this dir> python -m pip install ...

Falls back to the original resolver on any DoH failure, never breaks IP
literals or localhost, and caches by name for the TTL AliDNS reports.
"""
import json, socket, time, urllib.request

_orig = socket.getaddrinfo
_cache = {}
_DOH = "https://223.5.5.5/resolve?name=%s&type=A"


def _is_ip(host):
    try:
        socket.inet_aton(host); return True
    except OSError:
        return host.count(":") >= 2


def _doh(host):
    hit = _cache.get(host)
    if hit and hit[1] > time.time():
        return hit[0]
    req = urllib.request.Request(_DOH % host, headers={"User-Agent": "dohpatch"})
    d = json.load(urllib.request.urlopen(req, timeout=8))
    ans = [a for a in d.get("Answer", []) if a.get("type") == 1]
    ips = [a["data"] for a in ans]
    if not ips:
        raise OSError("no A record via DoH for %s" % host)
    ttl = min(a.get("TTL", 60) for a in ans)
    _cache[host] = (ips, time.time() + max(30, ttl))
    return ips


def getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if not host or host in ("localhost", "127.0.0.1", "::1") or _is_ip(host) \
            or host.endswith(".local"):
        return _orig(host, port, family, type, proto, flags)
    try:
        ips = _doh(host)
    except Exception:
        return _orig(host, port, family, type, proto, flags)
    out = []
    for ip in ips:
        out += _orig(ip, port, family or socket.AF_INET, type, proto, flags)
    return out


socket.getaddrinfo = getaddrinfo
