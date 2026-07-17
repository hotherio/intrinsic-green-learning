"""Variable-projection solver verification suite.

Verifies the convergence, scalability, determinism, and quality claims about
IGL's VP inner/outer loops: envelope-gradient equivalence of iterative inner
solvers (E1/E5), the direct-vs-Krylov-vs-SGD cost crossover (E2),
conditioning and tolerance transfer under target whitening (E3), outer-loop
acceleration (E4), end-metric invariance (E6), and generalized VP with a true
cross-entropy inner problem (E7). Each experiment pre-registers its
predictions in the module docstring and emits per-prediction verdicts into
its result JSON. Findings are consolidated in ``REPORT.md``.
"""
