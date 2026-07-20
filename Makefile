PYTHONPATH := src

test:
	PYTHONPATH=$(PYTHONPATH) python3 -m unittest discover -s tests -v

train:
	PYTHONPATH=$(PYTHONPATH) python3 -m mib_solution.train --train-pdfs data/train --labels data/train_labels.csv --artifacts artifacts --cache artifacts/render-cache

