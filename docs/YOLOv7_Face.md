# YOLOv7-Face usage

**NOTE**: The yaml file is not required.

* [Convert model](#convert-model)
* [Edit the config_infer_primary_yoloV7_face file](#edit-the-config_infer_primary_yolov7_face-file)

##

### Convert model

#### 1. Download the YOLOv7-Face repo and install the requirements

```
git clone https://github.com/derronqi/yolov7-face.git
cd yolov7-face
pip3 install -r requirements.txt
pip3 install onnx onnxsim onnxruntime
```

**NOTE**: It is recommended to use Python virtualenv.

#### 2. Copy conversor

Copy the `export_yoloV7_face.py` file from `DeepStream-Yolo-Face/utils` directory to the `yolov7-face` folder.

#### 3. Download the model

Download the `pt` file from [YOLOv7-Face](https://github.com/derronqi/yolov7-face) repo.

**NOTE**: You can use your custom model.

#### 4. Convert model

Generate the ONNX model file (example for YOLOv7-Face)

```
python3 export_yoloV7_face.py -w yolov7-face.pt --dynamic
```

**NOTE**: To change the inference size (defaut: 640)

```
-s SIZE
--size SIZE
-s HEIGHT WIDTH
--size HEIGHT WIDTH
```

Example for 1280

```
-s 1280
```

or

```
-s 1280 1280
```

**NOTE**: To simplify the ONNX model (DeepStream >= 6.0)

```
--simplify
```

**NOTE**: To use dynamic batch-size (DeepStream >= 6.1)

```
--dynamic
```

**NOTE**: To use explicit batch-size (example for batch-size = 4)

```
--batch 4
```

#### 5. Copy generated files

Copy the generated ONNX model file to the `DeepStream-Yolo-Face` folder.

##

### Edit the config_infer_primary_yoloV7_face file

Edit the `config_infer_primary_yoloV7_face.txt` file according to your model (example for YOLOv7-Face)

```
[property]
...
onnx-file=yolov7-face.onnx
...
```

**NOTE**: The **YOLOv7-Face** resizes the input with center padding. To get better accuracy, use

```
[property]
...
maintain-aspect-ratio=1
symmetric-padding=1
...
```
