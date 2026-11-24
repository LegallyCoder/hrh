PY ?= python3
.PHONY: test figures eval law paper

test:
	$(PY) -m pytest -q

figures:
	$(PY) run_hrh.py figures --paper paper

eval:
	$(PY) run_hrh.py eval

law:
	$(PY) run_hrh.py proxies && $(PY) run_hrh.py law && $(PY) run_hrh.py predict

paper:
	cd paper && pdflatex -interaction=nonstopmode main >/dev/null && bibtex main >/dev/null && pdflatex -interaction=nonstopmode main >/dev/null && pdflatex -interaction=nonstopmode main >/dev/null && echo "-> paper/main.pdf"
