PY ?= python3
.PHONY: test figures eval law

test:
	$(PY) -m pytest -q

figures:
	$(PY) run_hrh.py figures --paper paper

eval:
	$(PY) run_hrh.py eval

law:
	$(PY) run_hrh.py proxies && $(PY) run_hrh.py law && $(PY) run_hrh.py predict
