PYTHON ?= python3
DESIGN := recipes/synthetic-linear-dynamics/design.expected.json

.PHONY: help install-dev test test-core test-skill validate-design recipe manifest json compile check

help:
	@echo "JEPA Anything repository checks"
	@echo "  make install-dev      Install the core package and test tools"
	@echo "  make test             Run core and Skill regression tests"
	@echo "  make validate-design  Validate the reference compiled design"
	@echo "  make recipe           Run the no-training synthetic structural smoke audit"
	@echo "  make manifest         Verify metadata boundaries and local artifact integrity"
	@echo "  make check            Run every deterministic repository check"

install-dev:
	$(PYTHON) -m pip install -e './jepa-anything-core[dev]'

test: test-core test-skill

test-core:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest -p no:capture jepa-anything-core/tests -q

test-skill:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest -p no:capture jepa-anything-skill/tests -q

validate-design:
	$(PYTHON) jepa-anything-skill/scripts/validate_design.py $(DESIGN) >/dev/null

recipe:
	$(PYTHON) recipes/synthetic-linear-dynamics/run_recipe.py --quiet

manifest:
	$(PYTHON) checkpoints/verify_manifest.py --quiet

json:
	$(PYTHON) -m json.tool $(DESIGN) >/dev/null
	$(PYTHON) -m json.tool recipes/synthetic-linear-dynamics/recipe.json >/dev/null
	$(PYTHON) -m json.tool checkpoints/manifest.json >/dev/null
	$(PYTHON) -m json.tool checkpoints/manifest.schema.json >/dev/null

compile:
	$(PYTHON) -m compileall -q jepa-anything-core/src jepa-anything-skill/scripts recipes checkpoints

check: json compile test validate-design recipe manifest
