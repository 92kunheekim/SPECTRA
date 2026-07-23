# Load test

Standard-library load-tester for the live SPECTRA endpoint (no dependencies).

```bash
python deploy/loadtest/load_test.py https://<workspace>--spectra-tcr-pmhc-fastapi-app.modal.run
```

Reports **cold-start** latency, **warm** single-request percentiles
(p50/p95/p99), and **throughput** (req/s) under concurrency. The server also
exposes its own view at `/stats` (JSON) and `/metrics` (Prometheus text).
