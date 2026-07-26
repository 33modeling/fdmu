# Claude entry point

Read [AGENTS.md](AGENTS.md) before changing code or launching an experiment.
It is the environment-neutral operating contract for local and cluster agents.

Then choose exactly one runbook:

- RTX 4090 x2: [local_run/README.md](local_run/README.md)
- H100 fleet: [docs/CLUSTER_FLEET_RUNBOOK.md](docs/CLUSTER_FLEET_RUNBOOK.md)
- CPU evidence/LaTeX: [docs/FINAL_RESULTS_RUNBOOK.md](docs/FINAL_RESULTS_RUNBOOK.md)
- Documentation map: [docs/README.md](docs/README.md)
- Runtime logs and intermediate results:
  [docs/LLM_RUN_DIAGNOSTICS.md](docs/LLM_RUN_DIAGNOSTICS.md)

Do not infer commands from deleted historical plans or Git history. Check the
current scripts and configs, run preflight, and preserve every freeze/seal.
