FILES ?= app.py models/schemas.py services/email_service.py

.PHONY: format check

format:
	python -m black $(FILES)

check:
	python -m black --check $(FILES)
	python -m compileall $(FILES)
