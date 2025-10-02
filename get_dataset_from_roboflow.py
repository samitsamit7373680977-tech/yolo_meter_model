from roboflow import Roboflow


rf = Roboflow(api_key="z8Qwmhrpod5R1ExDHXUe")
project = rf.workspace("eb-meter").project("full_meter-po7qa")
version = project.version(2)
dataset = version.download("yolov8")