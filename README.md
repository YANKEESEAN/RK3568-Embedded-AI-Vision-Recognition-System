# RK3568 Embedded AI Vision Recognition System

## Introduction

This project is an embedded AI vision system developed on the RK3568 platform. It integrates LCD display, touchscreen interaction, camera acquisition, and AI-based visual recognition into a complete intelligent vision application.

The system leverages Embedded Linux, RKNN Toolkit2, and the RK3568 NPU accelerator to achieve real-time object detection and image recognition.

---

## System Architecture

```text
Camera
   │
   ▼
Image Capture (V4L2)
   │
   ▼
Image Preprocessing
   │
   ▼
RKNN Runtime
   │
   ▼
YOLOv8 Inference
   │
   ▼
Detection Result
   │
   ├── LCD Display
   ├── Touch Interaction
   └── Video Recording
```

---

## Features

### LCD Display

- BMP image rendering
- JPG image rendering
- PNG image rendering
- Image scaling
- Double-buffer rendering
- Detection box visualization

### Touchscreen Interaction

- Single-touch detection
- Coordinate mapping
- Region-based interaction
- Page navigation
- Swipe gesture recognition

### Camera Module

- USB/MIPI camera support
- Real-time preview
- Snapshot capture
- MJPEG video recording
- Local storage and playback

### AI Vision Recognition

- RKNN model deployment
- YOLOv8 object detection
- MobileNet image classification
- NPU-accelerated inference
- Real-time bounding boxes
- Confidence score display

### Extended Features

- PNG transparency rendering
- Image cache mechanism
- Photo album management
- Thumbnail preview
- Dataset annotation
- Buzzer alarm linkage

---

## Development Environment

### Hardware

- RK3568 Development Board
- LCD Touchscreen
- USB Camera

### Software

- Ubuntu 24.04 (WSL2)
- VSCode
- MobaXterm
- GCC / G++
- RKNN Toolkit2
- RKNN Runtime
- OpenCV
- V4L2
- Linux DRM

---

## Project Structure

```text
project/
│
├── src/
│   ├── yolov8_detect.cpp
│   ├── yolov8_wrapper.cpp
│   └── main.c
│
├── model/
│   ├── yolov8.rknn
│   └── coco_80_labels_list.txt
│
├── obj/
│
├── include/
│
├── material/
│
├── lib/
│
├── bin/
│
├── 3rdparty/
│
├── Makefile
│
└── README.md
```

---

## Technologies Used

- Embedded Linux
- V4L2 Framework
- DRM Framework
- RKNN Toolkit2
- RK3568 NPU
- YOLOv8
- OpenCV
- Multi-thread Programming
- Double Buffer Rendering

---

## Highlights

- RK3568 NPU Acceleration
- Real-Time YOLOv8 Detection
- Embedded Linux Development
- Multi-thread Architecture
- Touchscreen Human-Computer Interaction
- Camera and AI Collaborative Processing

---

## Demo

The system supports:

- Real-time camera preview
- Object detection visualization
- Category label display
- Confidence score display
- Snapshot capture
- Video recording

---

# RK3568嵌入式AI视觉识别系统

## 项目简介

本项目基于 RK3568 嵌入式 AI 开发板开发，实现了集 LCD 显示、触摸屏交互、摄像头采集以及 AI 视觉识别于一体的嵌入式智能视觉系统。

系统采用 Linux 嵌入式开发环境，结合 RKNN Toolkit2 与 RK3568 NPU 硬件加速能力，实现实时目标检测与图像识别功能，可广泛应用于智能安防、工业检测、智能终端等场景。

---

## 系统架构

```text
Camera
   │
   ▼
Image Capture (V4L2)
   │
   ▼
Image Preprocessing
   │
   ▼
RKNN Runtime
   │
   ▼
YOLOv8 Inference
   │
   ▼
Detection Result
   │
   ├── LCD Display
   ├── Touch Interaction
   └── Video Recording
```

---

## 主要功能

### LCD界面显示

- BMP 图片显示
- JPG 图片显示
- PNG 图片显示
- 图像缩放显示
- 双缓冲刷新
- 检测框与识别结果叠加显示

### 触摸屏交互

- 单点触摸
- 坐标映射
- 区域点击检测
- 页面切换
- 滑屏手势识别

### 摄像头功能

- USB/MIPI 摄像头采集
- 实时视频预览
- 图片抓拍
- MJPEG 视频录制
- 本地存储与回放

### AI视觉识别

- RKNN模型部署
- YOLOv8目标检测
- MobileNet图像分类
- NPU硬件加速推理
- 实时检测框绘制
- 类别与置信度显示

### 扩展功能

- PNG透明图标渲染
- 图片缓存机制
- 相册管理
- 缩略图预览
- 数据标注
- 蜂鸣器联动报警

---

## 开发环境

### Hardware

- RK3568 Development Board
- LCD Touch Screen
- USB Camera

### Software

- Ubuntu 24.04 (WSL2)
- VSCode
- MobaXterm
- GCC / G++
- RKNN Toolkit2
- RKNN Runtime
- OpenCV
- V4L2
- Linux DRM

---

## 项目目录

```text
project/
│
├── src/
│   ├── yolov8_detect.cpp
│   ├── yolov8_wrapper.cpp
│   └── main.c
│
├── model/
│   ├── yolov8.rknn
│   └── coco_80_labels_list.txt
│
├── obj/
│
├── include/
│
├── material/
│
├── lib/
│
├── bin/
│
├── 3rdparty/
│
├── Makefile
│
└── README.md
```

---

## 核心技术

- Embedded Linux
- V4L2 Video Framework
- DRM Display Framework
- RKNN Toolkit2
- RK3568 NPU
- YOLOv8
- OpenCV
- Multi-thread Programming
- Double Buffer Rendering

---

## 运行效果

系统可实现：

- 实时摄像头画面显示
- 目标检测框绘制
- 类别名称显示
- 置信度显示
- 触摸交互控制
- 视频录制与抓拍

---

## 项目亮点

- RK3568 NPU硬件加速
- YOLOv8实时目标检测
- 多线程架构设计
- 双缓冲无闪屏显示
- 摄像头与AI推理协同运行
- 支持图片、视频、检测结果统一管理

---

## License

This project is for educational and academic purposes only.
