PYTHON_VERSION := 3.12
VENV := $(PWD)/.venv
PYTHON := $(VENV)/bin/python$(PYTHON_VERSION)
PIP := $(PYTHON) -m pip
VERSION := $(file < VERSION)
PACKAGE := 'hother'
FORMAT := "md"  # License format

.PHONY: help lint test package clean install

build: clean venv install ### Builds the environment

build-dev: build install-dev ### Builds the environment with test dependencies

venv: clean-all ### Install a Python Virtual Environment.
	uv venv
	. $(VENV)/bin/activate


test: clean-build install-test  # Runs all the project tests
	$(PYTHON) -m pytest tests/

coverage:
	uv run $(PYTHON) -m pytest src --cov=hother

vulture: # Reports likely dead code in src/
	uv run vulture src/ --min-confidence 80

package: clean # Runs the project setup
	hatch build



clean: clean-build clean-pyc ### Removes environment and artifacts

clean-all: clean clean-env ### Removes all

clean-build: ### Removes builds
	find . -type d -iname "build" ! -path "./.venv/*" -exec rm -rf {} +
	find . -type d -iname "dist" ! -path "./.venv/*" -exec rm -rf {} +
	find . -type d -iname "*.egg-info" ! -path "./.venv/*" -exec rm -rf {} +

clean-env: ### Removes environment directory
	rm -rf $(VENV) &> /dev/null
###. $(VENV)/bin/deactivate &> /dev/null

clean-pyc: ### Removes python compiled bytecode files
	find . -iname "*.pyc" ! -path "./.venv/*" -delete
	find . -type d -iname "__pycache__" ! -path "./.venv/*" -exec rm -rf {} +



build-wheel:
	hatch build -t wheel

version:
	@echo "Current version: $(shell hatch version 2>/dev/null || echo 'No tags yet')"

tag-release:
	@echo "Run: git tag -a v$(VERSION) -m 'Release v$(VERSION)' && git push origin v$(VERSION)"

install: # Installs required dependencies
	uv sync

install-dev: # Installs required dependencies
	uv sync --group dev
	. $(VENV)/bin/activate
	lefthook install


install-docs:
	uv venv
	uv sync --group doc
	. $(VENV)/bin/activate

install-dist:	### Installs specific distribution
	$(PIP) install dist/$(PACKAGE)-$(VERSION).tar.gz

install-wheel:
	$(PIP) install dist/$(PACKAGE)-$(VERSION)-py3-none-any.whl




docs-publish:  # Publishes the documentation
	$(PIP) install .
	mike deploy --push --update-aliases $(VERSION) latest


licenses:
	uvx --from pip-licenses==5.0.0 pip-licenses --from=mixed --order count -f $(FORMAT) --output-file licenses.$(FORMAT)

changelog: ### Generate full changelog
	git-cliff -o CHANGELOG.md

changelog-unreleased: ### Preview unreleased changes
	@git-cliff --unreleased

changelog-tag: ### Get changelog for latest tag (for releases)
	@git-cliff --latest --strip header

# --- geometry benchmark suite (manual; see benchmarks/geometry/README.md) ---
.PHONY: geometry-smoke geometry-tier0
geometry-smoke: ### run every geometry block on a tiny setting (under a minute each)
	for b in r0_rank_go_nogo g0_geometry g1_rho_law g2_gauge g4_oracle_floor; do uv run python -m benchmarks.geometry.blocks.$$b --smoke || exit 1; done

geometry-tier0: ### the analytic blocks of the geometry suite, one after the other (R0, G0, G1, G2, G4)
	for b in r0_rank_go_nogo g0_geometry g1_rho_law g2_gauge g4_oracle_floor; do uv run python -m benchmarks.geometry.blocks.$$b || exit 1; done

.PHONY: geometry-scale-smoke geometry-w1 geometry-s0 geometry-s1 geometry-s2 geometry-s3
geometry-scale-smoke: ### run every scale block (W1, S0-S3) on the synthetic stand-in, CPU, under a minute
	for b in w1_gate s0_baselines s1_anchor_budget s2_penalty_at_scale s3_learning_curve; do uv run python -m benchmarks.geometry.blocks.$$b --smoke --no-gate || exit 1; done

SETS ?= mnist
geometry-w1: ### the week-1 gate on $(SETS) (noise floor, frozen PCA chart, inner-solve and ridge checks)
	IGL_BENCH_LABEL=scale uv run python -m benchmarks.geometry.blocks.w1_gate --sets $(SETS) --device $${IGL_BENCH_DEVICE:-mps}
geometry-s0: ### the baselines of $(SETS)
	IGL_BENCH_LABEL=scale uv run python -m benchmarks.geometry.blocks.s0_baselines --sets $(SETS) --k 16,64 --device $${IGL_BENCH_DEVICE:-mps}
geometry-s1: ### the anchor budget of $(SETS)
	IGL_BENCH_LABEL=scale uv run python -m benchmarks.geometry.blocks.s1_anchor_budget --sets $(SETS) --k 16,64 --device $${IGL_BENCH_DEVICE:-mps}
geometry-s2: ### the penalty ablation of $(SETS) at R_BIND (from S1) and 256
	IGL_BENCH_LABEL=scale uv run python -m benchmarks.geometry.blocks.s2_penalty_at_scale --sets $(SETS) --anchors $(R_BIND),256 --control --device $${IGL_BENCH_DEVICE:-mps}
geometry-s3: ### the learning curve of $(SETS) at R_BIND
	IGL_BENCH_LABEL=scale uv run python -m benchmarks.geometry.blocks.s3_learning_curve --sets $(SETS) --anchors $(R_BIND) --device $${IGL_BENCH_DEVICE:-mps}
