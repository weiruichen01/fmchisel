.PHONY: test checkstyle


all: checkstyle test

# Command to run pytest for correctness tests
test:
	python3 -m pytest --disable-warnings --ignore=test/data/ test/

# Command to run ruff for linting and formatting code
checkstyle:
	pre-commit run --all-files  -v 
