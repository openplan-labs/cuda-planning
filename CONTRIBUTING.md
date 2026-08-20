# Contributing

Thanks for looking into cuda-planning — contributions are welcome, and
you do not need a GPU for most of them.

## Development setup

```bash
git clone https://github.com/openplan-labs/cuda-planning.git
cd cuda-planning
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
```

That installs the CPU reference backend, pytest, ruff, and the
benchmark dependencies. Run the checks the CI runs:

```bash
ruff check .
pytest
```

## Working without a GPU

The library is designed so the CPU path carries the full test suite:

- Every algorithm has a vectorized NumPy reference implementation, and
  every behavioural test runs against it unconditionally.
- CUDA tests are additional equivalence checks (`CPU == CUDA`) that
  skip automatically when no device is present — you will see them as
  `s` in the pytest output. CI runs the CPU suite on every push.
- Kernel sources are plain CUDA C in `cuplan/kernels/*.cu`. They are
  compiled by NVRTC, which needs CuPy but not a device, so
  `pip install 'cupy-cuda12x[ctk]'` lets you at least compile-check a
  kernel change (`pytest tests/test_kernels.py`).

If you change a kernel, change the matching NumPy reference (or vice
versa) in the same commit — the equivalence tests are the contract
between them. A PR that touches only one side of the pair will fail on
a GPU machine even if CI is green, so say in the PR description
whether you ran the CUDA suite.

## With a GPU

`pip install -e '.[dev,cuda12]'` (or `cuda12` → `cuda11` for older
drivers). If CuPy reports missing CUDA headers, use
`pip install 'cupy-cuda12x[ctk]'` — the `ctk` extra ships the headers
as pip wheels, no toolkit install or sudo required. Then run

```bash
pytest            # CUDA equivalence tests now execute
python -m cuplan.benchmark --out /tmp/bench   # sanity-check performance
```

## Style

- `ruff check .` and `ruff format` are the arbiters of style.
- Public functions are typed and carry docstrings in the imperative
  mood ("Return the plan", not "Returns the plan").
- Cite the paper the first time an algorithm appears in a module —
  name and year in the docstring, full reference in the module header.
- Benchmarks come with conditions: machine, problem set, and seed, or
  they do not get committed.

## Commits and PRs

- Keep commits focused; explain the *why* in the body if it is not
  obvious.
- New algorithms should follow the existing shape: one module, a CPU
  reference, a CUDA path, equivalence tests, and a docs page stating
  the semantics and the parallelization strategy. Open an issue first
  for anything on the [roadmap](https://openplan-labs.github.io/cuda-planning/roadmap/)
  so we can agree on the approach before you invest in kernels.
