SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = axion
BUILDDIR      = build

clean:
	find . -name '*egg-info' | xargs rm -rf
	find . -name '.benchmarks' | xargs rm -rf
	find . -name '.coverage' | xargs rm -rf
	find . -name '.mypy_cache' | xargs rm -rf
	find . -name '.pyre' | xargs rm -rf
	find . -name '.pytest_cache' | xargs rm -rf
	find . -name '.tox' | xargs rm -rf
	find . -name '__pycache__' | xargs rm -rf
	find . -name 'reports' | xargs rm -rf
	find . -name '.ruff_cache' | xargs rm -rf
	find . -name '*.pyc' -delete 2>&1
	find . -name '*.pyo' -delete 2>&1

prerequisites:
	python -m pip install -U pip setuptools wheel setuptools_scm[toml]

install:prerequisites
	python -m pip install -U --upgrade-strategy eager '.[all]'

develop:prerequisites
	python -m pip install -U --upgrade-strategy eager -e '.[all]'

lint:
	pre-commit run --all-files --hook-stage manual

package:prerequisites
	python -m pip install setuptools>=40.8.0 wheel setuptools_scm[toml]>=6.0
	python -m pip install -U --upgrade-strategy eager build
	pyproject-build --no-isolation

test: install
	python -m pip install pytest coverage fastapi fastapi httpx uvicorn requests
	python -m pip -U openssl
	coverage run -m pytest --junitxml=pytest_report.xml
	coverage xml -o coverage.xml
tox:
	python -m pip install tox
	python -m tox

publish:
	mkdocs gh-deploy
