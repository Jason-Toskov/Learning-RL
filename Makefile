SHELL = /bin/bash

venv:
	test -d venv || python3 -m venv venv

install_core: venv
	source venv/bin/activate && \
	python3 -m pip install --upgrade "pip==23.1.2" && \
	python3 -m pip install -e .
