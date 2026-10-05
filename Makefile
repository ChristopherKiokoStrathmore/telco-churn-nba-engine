.PHONY: install test train serve export-demo

install:
	pip install -r requirements.txt

test:
	pytest

train:
	PYTHONPATH=src python -m telco_nba.train

serve:
	PYTHONPATH=src uvicorn telco_nba.api:app --host localhost --port 8000

export-demo:
	PYTHONPATH=src python scripts/export_web_demo.py
