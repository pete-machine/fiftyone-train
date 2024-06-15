from collections import Counter
from datetime import datetime
import os
from pathlib import Path
import shutil


from clearml import Task
from ultralytics import YOLO


os.environ["FIFTYONE_DATABASE_URI"] = "mongodb://root:example@192.168.100.120:27017"

# Load a dataset
dataset_name = 'christmas-trees'
path_output_path = Path.home() / "exported_data" / dataset_name
reload_dataset = False

now_str = datetime.now().strftime("%Y_%m_%dT%H_%M_%S")
task = Task.init(project_name="ChistmasTrees", task_name=f"{now_str}")

if reload_dataset:
    import fiftyone as fo
    import fiftyone.utils.random as four
    from fiftyone import ViewField as F


    # Load a dataset
    datasets = fo.list_datasets()
    print(f"Datasets: {fo.list_datasets()}")

    assert dataset_name in datasets, f"Dataset {dataset_name} not found in {datasets}"
    dataset_full = fo.load_dataset(dataset_name)

    def flatten_extend(list_of_lists):
        flat_list = []
        for row in list_of_lists:
            flat_list.extend(row)
        return flat_list

    dataset = dataset_full.select_fields("label").match(F("label").exists())
    classes = sorted(Counter(flatten_extend(dataset.values("label.detections.label"))).keys())
    print("Dataset loaded!!")
    print(f"Found {len(dataset)} samples with ground truth labels")


    ## delete existing tags to start fresh
    dataset.untag_samples(dataset.distinct("tags"))

    splits_ratio = {"train": 0.8, "val": 0.2}
    four.random_split(dataset, splits_ratio)

    shutil.rmtree(path_output_path, ignore_errors=True)

    path_output_path.mkdir(parents=True, exist_ok=True)

    for split_name in splits_ratio:
        split = f"{split_name}"
        split_view = dataset.match_tags(split_name)
        split_view.export(
            export_dir=str(path_output_path),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field= "label",
            classes=classes,
            split=split)

path_dataset_yaml = path_output_path / "dataset.yaml"
model_variant = "yolov8x"  #yolov8n.pt yolov8s.pt yolov8m.pt yolov8l.pt yolov8x.pt
task.set_parameter("model_variant", model_variant)
model = YOLO(f"{model_variant}.pt")

configs = dict(data=str(path_dataset_yaml), epochs=1000, imgsz=2048, batch=6, device=0, patience=0)
task.connect(configs)

results = model.train(**configs)
# subprocess.run(["yolo", "task=detect", "mode=train", "model=yolov8n.pt", "data=birds_train/dataset.yaml", "epochs=60", "imgsz=640", "batch=16"], capture_output=True)
# # !yolo task=detect mode=train model=yolov8n.pt data=birds_train/dataset.yaml epochs=60 imgsz=640 batch=16