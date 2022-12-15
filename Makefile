install:
	@pip install -e .
# @pip install . # for procduction

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -f */.ipynb_checkpoints
	@rm -Rf build
	@rm -Rf */__pycache__
	@rm -Rf */*.pyc
	@rm -Rf */*egg-info

all: install clean

run_train:
	python -c 'from modules.interface.main import train; train()'

run_api:
	@python modules/api/fast_api.py

test:
	@pytest -v tests
