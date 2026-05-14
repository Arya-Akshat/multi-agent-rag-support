.PHONY: setup run-api run-ui test clean

setup:
	bash scripts/setup.sh

run-api:
	PYTHONPATH=. .venv/bin/python3 api/main.py

run-ui:
	.venv/bin/streamlit run ui/streamlit_app.py

test:
	.venv/bin/pytest tests/

clean:
	rm -rf __pycache__ .pytest_cache logs/*.log
