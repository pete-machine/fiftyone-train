# Run model training


Install environment

    micromamba env create -f environment.yaml -y

Activate environment

    micromamba activate fiftyone-train


- Ensure that a mongdb server is running and set environment variable:

    os.environ["FIFTYONE_DATABASE_URI"] = "mongodb://root:example@192.168.100.120:27017"

- Ensure that user is logged in with clearml
- Set `reload_datase=True` 

