# Jetson Nano OpenCV CUDA Test
L4T (Linux for Tegra) OS used by Jetson Nano is already included with NVIDIA JetPack SDK. It bundels all jetson platform software, including TensorRT, cuDNN, CUDA Toolkit, VisionWorks, GStreamer, and OpenCV, all built on top of L4T with LTS Linux kernel. However, the default OpenCV is not CUDA enabled, wich means we can't use CUDA to accelerate for example DNN inferencing in OpenCV.

## Install OpenCV with CUDA Enable
To make CUDA enable in OpenCV, we need to build the OpenCV with CUDA from Source. You follow this repo to do that : https://github.com/mdegans/nano_build_opencv

## Test OpenCV with CUDA Enabled
- Run `opencv-info.py` to check build information (to make sure if CUDA already enabled in OPENCV),
```
$ python3 opencv-info.py
```
- Run `test-cuda.py` to test basic CUDA operation using OpenCV,
```
$ python3 test-cuda.py
```
- Run `opencv-dnn-cuda.py` to test CUDA as target and backend OpenCV DNN (Tiny Yolo V3 Inferencing @coco dataset),
    - Using CUDA as backend and target OpenCV DNN :
    ```
    $ python3 opencv-dnn-cuda.py --backend 'CUDA' --target 'CUDA'
    ```
    - Using OpenCV CPU as backend and target OpenCV DNN (just for comparison):
    ```
    $ python3 opencv-dnn-cuda.py --backend 'OPENCV' --target 'CPU'
    ```