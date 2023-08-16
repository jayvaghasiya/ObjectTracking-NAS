# Object Tracker for Pretrained and finetuned yolo models:
<div align="center">
<h4>Drone Object detection</h4>
<img src="https://github.com/jayvaghasiya/ObjectTracking-NAS/blob/main/assets/images/drone.gif" width="400"> 
<h4>Road-Object detection</h4> 
<img src="https://github.com/jayvaghasiya/ObjectTracking-NAS/blob/main/assets/images/road.gif" width="400">
<h4>Tyre detection</h4><img src="https://github.com/jayvaghasiya/ObjectTracking-NAS/blob/main/assets/images/tyre.gif" width="400">
<h4>Pracel detection</h4>
<img src="https://github.com/jayvaghasiya/ObjectTracking-NAS/blob/main/assets/images/parcel.gif" width="400">
</div>

<div align="center">
<h3>Botsort Tracking</h3> 
<img src="https://github.com/jayvaghasiya/ObjectTracking-NAS/blob/main/assets/images/BotSort.drawio.png" alt="botsort">
  
</div>


## Introduction

This repo contains a collections of pluggable state-of-the-art multi-object trackers for object detectors.



<details>
<summary>Supported tracking methods</summary>

| Trackers | HOTA↑ | MOTA↑ | IDF1↑ |
| -------- | ----- | ----- | ----- |
| [OCSORT](https://github.com/noahcao/OC_SORT)[](https://arxiv.org/abs/2203.14360) | | | |
| [ByteTrack](https://github.com/ifzhang/ByteTrack)[](https://arxiv.org/abs/2110.06864) | | | |
| [DeepOCSORT](https://arxiv.org/abs/2302.11813) | | | |
| [BoTSORT](https://arxiv.org/abs/2206.14651) | | | |
| [StrongSORT](https://github.com/dyhBUPT/StrongSORT) | | | |



</details>



## Why using this tracking toolbox?

Everything is designed with simplicity and flexibility in mind. We don't hyperfocus on results on a single dataset, we prioritize real-world results. If you don't get good tracking results on your custom dataset with the out-of-the-box tracker configurations, use the `examples/evolve.py` script for tracker hyperparameter tuning.

## Installation

Start with [**Python>=3.8**](https://www.python.org/) environment.

If you want to run the YOLOv8, YOLO-NAS or YOLOX examples:

```
git clone https://github.com/jayvaghasiya/ObjectTracking-NAS.git
pip install -v -e .
```

but if you only want to import the tracking modules you can simply:

```
pip install boxmot
```

## YOLOv8 | YOLO-NAS | YOLOX examples

<details>
<summary>Tracking</summary>
</details>
<details>
<summary>Yolo models</summary>



```bash
$ python examples/track.py --yolo-model yolov8n       # bboxes only
  python examples/track.py --yolo-model yolo_nas_s    # bboxes only
  python examples/track.py --yolo-model yolox_n       # bboxes only
                                        yolov8n-seg   # bboxes + segmentation masks
                                        yolov8n-pose  # bboxes + pose estimation

```

</details>

<details>
<summary>Tracking methods</summary>

```bash
$ python examples/track.py --tracking-method deepocsort
                                             strongsort
                                             ocsort
                                             bytetrack
                                             botsort
```

</details>

<details>
<summary>Tracking sources</summary>

Tracking can be run on most video formats

```bash
$ python examples/track.py --source 0                               # webcam
                                    img.jpg                         # image
                                    vid.mp4                         # video
                                    path/                           # directory
                                    path/*.jpg                      # glob
                                    'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                    'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

</details>

<details>
<summary>Select ReID model</summary>

Some tracking methods combine appearance description and motion in the process of tracking. For those which use appearance, you can choose a ReID model based on your needs from this [ReID model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO). These model can be further optimized for you needs by the [reid_export.py](https://github.com/mikel-brostrom/yolo_tracking/blob/master/boxmot/deep/reid_export.py) script

```bash
$ python examples/track.py --source 0 --reid-model lmbn_n_cuhk03_d.pt               # lightweight
                                                   osnet_x0_25_market1501.pt
                                                   mobilenetv2_x1_4_msmt17.engine
                                                   resnet50_msmt17.onnx
                                                   osnet_x1_0_msmt17.pt
                                                   clip_market1501.pt               # heavy
                                                   clip_vehicleid.pt
                                                   ...
```

</details>

<details>
<summary>Filter tracked classes</summary>

By default the tracker tracks all MS COCO classes.

If you want to track a subset of the classes that you model predicts, add their corresponding index after the classes flag,

```bash
python examples/track.py --source 0 --yolo-model yolov8s.pt --classes 16 17  # COCO yolov8 model. Track cats and dogs, only
```

[Here](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/) is a list of all the possible objects that a Yolov8 model trained on MS COCO can detect. Notice that the indexing for the classes in this repo starts at zero

</details>

<summary>For tracking Object with custom yolo-nas model</summary>

you need to pass two command line argumenst:

pass these argument ::

Exact name of the yolo model(Ex: yolo_nas_s,yolo_nas_l,yolo_nas_m):

--yolo-model yolo_nas_s

Path of the checkpoint :

--chekpoint_path ../tyre_model.pth

Enter the classes you want to trak with your custom model (In original sequence):

--custom_classes tyre parcel dummy

```bash
$ python examples/track.py --yolo-model yolo_nas_s --checkpoint_path ../tyre_model.pth --custom_classes tyre parcel car person 
```

</details>
