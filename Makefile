install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	isort *.py
	black *.py
	
lint:
	pylint *.py

all: install format lint
