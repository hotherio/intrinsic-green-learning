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
