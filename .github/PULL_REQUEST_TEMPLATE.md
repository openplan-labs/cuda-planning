# What this changes

<!-- One or two sentences. Link the issue if there is one. -->

## Checklist

- [ ] `ruff check .` and `pytest` pass locally
- [ ] Kernel changes come with the matching NumPy reference change
      (they are one contract — see CONTRIBUTING.md)
- [ ] I ran the CUDA test suite on a GPU machine, or noted here that
      I could not
- [ ] New algorithms cite their paper and document the
      parallelization strategy
- [ ] Benchmark numbers, if any, state machine, problem set, and seed
