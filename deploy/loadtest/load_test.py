#!/usr/bin/env python3
"""Load-test the live SPECTRA endpoint. Standard library only (no installs).

    python deploy/loadtest/load_test.py https://<workspace>--spectra-tcr-pmhc-fastapi-app.modal.run

Reports: cold-start latency, warm single-request percentiles (p50/p95/p99),
and throughput (req/s) under concurrency. Use these numbers verbatim on your
resume / README.
"""
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = (sys.argv[1] if len(sys.argv) > 1 else "").rstrip("/")
if not BASE:
    sys.exit("usage: python load_test.py <base-url>")
URL = BASE + "/predict"

PAYLOAD = json.dumps({
    "tra_seq": "AQTVTQSQPEMSVQEAETVTLSCTYDTSENNYYLFWYKQPPSRQMILVIRQEAYKQQNATENRFSVNFQKAAKSFSLKISDSQLGDTAMYFCALATHTGTASKLTFGTGTRLQVTL",
    "trb_seq": "ETGVTQTPRHLVMGMTNKKSLKCEQHLGHNAMYWYKQSAKKPLELMFVYSLEERVENNSVPSRFSPECPNSSHLFLHLHTLQPEDSALYLCASSQDPGSSYNEQFFGPGTRLTVLE",
    "peptide": "AAFKRSCLK",
    "mhc_seq": "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDQETRNVKAQSQTDRVDLGTLRGYYNQSEAGSHTIQIMYGCDVGSDGRFLRGYRQDAYDGKDYIALNEDLRSWTAADMAAQITKRKWEAAHEAEQLRAYLDGTCVEWLRRYLENGKETLQ",
}).encode()


def pctile(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    f = int(k)
    return xs[f] if f + 1 >= len(xs) else xs[f] + (xs[f + 1] - xs[f]) * (k - f)


def hit(_=None):
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(URL, data=PAYLOAD,
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            r.read()
            code = r.status
    except Exception as e:
        code = "ERR:" + type(e).__name__
    return (time.perf_counter() - t0) * 1000.0, code


def summarize(lat):
    return (f"p50 {pctile(lat,50):6.0f} ms   p95 {pctile(lat,95):6.0f} ms   "
            f"p99 {pctile(lat,99):6.0f} ms   mean {sum(lat)/len(lat):6.0f} ms")


def main():
    print(f"target: {URL}\n")
    cold, code = hit()
    print(f"[cold start]  first request: {cold:,.0f} ms  (status {code})")

    for _ in range(12):
        hit()                                   # warm up

    seq = [ms for ms, c in (hit() for _ in range(30)) if c == 200]
    print(f"\n[warm, sequential]  n={len(seq)}")
    print("  " + (summarize(seq) if seq else "no successful requests"))

    N, C = 120, 8
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=C) as ex:
        res = list(ex.map(hit, range(N)))
    wall = time.perf_counter() - t0
    lat = [ms for ms, c in res if c == 200]
    errs = [c for _, c in res if c != 200]
    print(f"\n[load]  {N} requests @ concurrency {C}")
    print(f"  throughput {N/wall:5.1f} req/s   wall {wall:.1f}s   errors {len(errs)}")
    print("  " + (summarize(lat) if lat else "no successful requests"))
    print("\nTip: right after this, open  <url>/stats  to see the server-side view.")


if __name__ == "__main__":
    main()
