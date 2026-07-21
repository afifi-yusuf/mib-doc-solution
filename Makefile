PYTHONPATH := src

test:
	PYTHONPATH=$(PYTHONPATH) python3 -m unittest discover -s tests -v

infer:
	PYTHONPATH=$(PYTHONPATH) python3 -m mib_solution.infer $(IN) $(OUT)

docker-build:
	docker build -t mib-submission .
