install:
	@python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt

lint:
	@pylint $(shell find . -name "*.py" -not -path "./.venv/*")

format:
	black --exclude .venv .

run:
	@python3 app.py

