.PHONY: setup install run clean

PYTHON = python3
VENV = discovera

setup:
	@echo "Setting up virtual environment..."
	@if [ ! -d "$(VENV)" ]; then $(PYTHON) -m venv $(VENV); fi
	@echo "Virtual environment created."

activate:
	@echo "Activating virtual environment..."
	@bash -i -c "source $(VENV)/bin/activate && exec $$SHELL"
	@echo "Virtual environment activated."

install:
	@source $(VENV)/bin/activate && pip install pipreqs
	@source $(VENV)/bin/activate && pipreqs . --force
	@source $(VENV)/bin/activate && pip install -r requirements.txt
	@echo "All dependencies installed."

run:
	@source $(VENV)/bin/activate && $(PYTHON) your_script.py

clean:
	rm -rf $(VENV)
	rm -f requirements.txt
	@echo "Environment cleaned."
