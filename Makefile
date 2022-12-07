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

test:
	@pytest -v tests
