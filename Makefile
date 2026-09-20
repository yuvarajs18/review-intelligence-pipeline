# Pipeline order: ingest -> validate -> score -> embed -> topics -> aggregate
# -> detect. Targets are independent (no Make prerequisites between them,
# matching the ingest/aggregate convention from Stage 1) - run them in
# order manually. `detect` will raise NotImplementedError until its bodies
# are filled in - see pipeline/detect.py.
.PHONY: ingest validate score embed topics aggregate detect test

PYTHON = .venv/bin/python

ingest:
	$(PYTHON) -m pipeline.ingest

validate:
	$(PYTHON) -m pipeline.validate

score:
	$(PYTHON) -m pipeline.score

embed:
	$(PYTHON) -m pipeline.embed

topics:
	$(PYTHON) -m pipeline.topics

aggregate:
	$(PYTHON) -m pipeline.aggregate

detect:
	$(PYTHON) -m pipeline.detect

dashboard:
	.venv/bin/streamlit run dashboard/app.py

test:
	$(PYTHON) -m pytest
