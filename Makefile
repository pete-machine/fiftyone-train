MICROMAMBA_RUN=micromamba run -n fiftyone-train

.PHONY: env-create
env-create:  ## Create python environment with micromamba
	micromamba env create -f environment.yaml -y


.PHONY: train
train:  ## Small regression test (<1 hour) using tiny-yolo and pascal voc
	$(MICROMAMBA_RUN) python main.py