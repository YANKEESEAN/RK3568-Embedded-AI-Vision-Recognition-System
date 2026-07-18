# YOLOv8 视觉识别接口调用实践指南

## 一、项目概览

本项目是一个基于 **RKNN（瑞芯微神经网络推理框架）** 的嵌入式视觉识别系统，核心功能是通过摄像头采集图像，使用 YOLOv8 模型进行实时目标检测，并将结果绘制到 LCD 屏幕上。

### 核心文件架构

```
camera/
├── include/
│   ├── yolov8_wrapper.h    # YOLOv8 C接口定义（核心API）
│   └── yolov8_detect.h     # 检测结果绘制接口
├── src/
│   ├── yolov8_wrapper.cpp  # YOLOv8 RKNN推理实现
│   ├── yolov8_detect.cpp   # 检测结果绘制实现
│   └── main.c              # 主程序（摄像头+LCD+交互）
└── model/
    ├── yolov8.rknn         # YOLOv8 RKNN模型文件
    └── coco_80_labels_list.txt  # 类别标签文件
```

---

## 二、YOLOv8 接口体系详解

### 2.1 核心数据结构

#### 检测结果结构体 `yolov8_detect_result_t`

```c
typedef struct {
    int x;              // 边界框左上角X坐标
    int y;              // 边界框左上角Y坐标
    int width;          // 边界框宽度
    int height;         // 边界框高度
    float confidence;   // 置信度（0~1，越大越可靠）
    int class_id;       // 类别ID（0~79，对应COCO数据集）
    char class_name[64]; // 类别名称（如"person"、"car"）
} yolov8_detect_result_t;
```

**示例**：检测到一个人

| 字段 | 值 | 说明 |
|------|-----|------|
| x | 100 | 框左上角在屏幕上的X位置 |
| y | 80 | 框左上角在屏幕上的Y位置 |
| width | 50 | 框的宽度 |
| height | 120 | 框的高度 |
| confidence | 0.95 | 95%置信度，非常可靠 |
| class_id | 0 | 类别ID为0，表示"person" |
| class_name | "person" | 类别名称 |

#### 检测结果列表 `yolov8_result_list_t`

```c
typedef struct {
    int count;                              // 检测到的目标数量
    yolov8_detect_result_t results[128];    // 最多128个检测结果
} yolov8_result_list_t;
```

#### 上下文句柄 `yolov8_context_t`

```c
typedef void* yolov8_context_t;  // YOLOv8模型的上下文句柄
```

这个句柄类似于一把"钥匙"，在初始化时获得，之后所有操作都需要携带它。

---

### 2.2 核心API接口

YOLOv8 接口遵循 **"初始化 → 使用 → 释放"** 的标准模式：

| 接口 | 功能 | 参数 | 返回值 |
|------|------|------|--------|
| `yolov8_init()` | 初始化YOLOv8模型 | 模型路径、标签路径 | 上下文句柄（成功）/ NULL（失败） |
| `yolov8_release()` | 释放YOLOv8模型资源 | 上下文句柄 | 无 |
| `yolov8_detect_rgb()` | 对RGB图像进行检测 | 上下文、RGB数据、宽高、结果输出 | 0（成功）/ -1（失败） |
| `yolov8_detect_and_draw()` | 检测并直接绘制到屏幕 | 上下文、RGB数据、宽高 | 0（成功）/ -1（失败） |

---

## 三、如何在自己的代码中调用 YOLOv8

### 3.1 集成步骤概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        完整调用流程                              │
├─────────────────────────────────────────────────────────────────┤
│  1. 包含头文件                                                   │
│     ↓                                                            │
│  2. 声明YOLOv8上下文句柄                                         │
│     ↓                                                            │
│  3. 设备初始化时调用 yolov8_init()                                │
│     ↓                                                            │
│  4. 图像处理循环中调用 yolov8_detect_rgb() 或 yolov8_detect_and_draw() │
│     ↓                                                            │
│  5. 设备退出时调用 yolov8_release()                               │
└─────────────────────────────────────────────────────────────────┘
```

---

### 3.2 详细实现步骤

#### 步骤1：包含头文件

在你的 `.c` 文件顶部添加：

```c
#include "yolov8_wrapper.h"
#include "yolov8_detect.h"
```

#### 步骤2：声明上下文句柄

在全局变量区域声明：

```c
yolov8_context_t yolov8_ctx = NULL;  // YOLOv8上下文，初始化为空
```

#### 步骤3：初始化模型（在设备初始化函数中）

```c
// 初始化YOLOv8模型
yolov8_ctx = yolov8_init("./model/yolov8.rknn", "./model/coco_80_labels_list.txt");
if (yolov8_ctx == NULL) {
    printf("YOLOv8模型初始化失败！检测功能将被禁用。\n");
} else {
    printf("YOLOv8模型初始化成功！\n");
}
```

**参数说明**：
- 第一个参数：RKNN模型文件路径（`.rknn`格式）
- 第二个参数：类别标签文件路径（每行一个类别名称）

#### 步骤4：执行检测（两种方式）

**方式A：仅检测，不绘制（推荐用于后台处理）**

```c
// 准备RGB图像数据（必须是RGB888格式，即每个像素3字节：R, G, B）
unsigned char rgb_data[640 * 480 * 3];  // 640x480的RGB图像

// 声明检测结果变量
yolov8_result_list_t results;

// 执行检测
int ret = yolov8_detect_rgb(yolov8_ctx, rgb_data, 640, 480, &results);

if (ret == 0 && results.count > 0) {
    // 遍历检测结果
    for (int i = 0; i < results.count; i++) {
        yolov8_detect_result_t *r = &results.results[i];
        printf("检测到: %s (%.1f%%) @ (%d, %d, %d, %d)\n",
               r->class_name, r->confidence * 100,
               r->x, r->y, r->width, r->height);
    }
}
```

**方式B：检测并直接绘制到LCD屏幕**

```c
// 准备RGB图像数据
unsigned char rgb_data[640 * 480 * 3];

// 执行检测并绘制（内部会自动调用yolov8_detect_rgb和yolov8_draw_results）
yolov8_detect_and_draw(yolov8_ctx, rgb_data, 640, 480);
```

#### 步骤5：释放资源（在设备退出函数中）

```c
if (yolov8_ctx != NULL) {
    yolov8_release(yolov8_ctx);
    yolov8_ctx = NULL;
}
```

---

## 四、项目中的实际应用案例

### 4.1 main.c 中的完整调用流程

#### 初始化阶段

```c
// 在 dev_init() 函数中
yolov8_ctx = yolov8_init("./model/yolov8.rknn", "./model/coco_80_labels_list.txt");
if (yolov8_ctx == NULL) {
    printf("YOLOv8 model initialization failed!\n");
} else {
    printf("YOLOv8 model initialized successfully!\n");
}
```

#### 实时检测阶段

```c
void *read_camera_data(void *arg) {
    while (1) {
        if (camera_flag == 1) {
            // 1. 获取摄像头帧
            camera_capture_get_frame(&jpeg_buf);
            
            // 2. 如果开启检测模式，先转换为RGB
            if (detect_flag == 1 && yolov8_ctx != NULL) {
                yuyv_to_rgb888((unsigned char *)jpeg_buf.start, rgb_buf, 640, 480, 640, 480);
            }
            
            // 3. 转换为JPEG并显示
            yuyv2jpeg(jpeg_buf.start, jpeg_buf.length, 99);
            lcd_draw_jpeg(0, 0, NULL, jpeg_buf.start, jpeg_buf.length, 1);
            
            // 4. 执行检测并绘制结果
            if (detect_flag == 1 && yolov8_ctx != NULL) {
                yolov8_detect_and_draw(yolov8_ctx, rgb_buf, 640, 480);
            }
        }
    }
}
```

#### 释放阶段

```c
// 在 dev_uninit() 函数中
if (yolov8_ctx != NULL) {
    yolov8_release(yolov8_ctx);
    yolov8_ctx = NULL;
}
```

---

### 4.2 数据流完整链路

```
摄像头采集(YUYV格式)
        ↓
yuyv_to_rgb888() 转换为RGB888
        ↓
yolov8_detect_rgb() 执行RKNN推理
        ↓
┌───────────────────────────────┐
│  RKNN推理内部流程:            │
│  1. letterbox缩放图像          │
│  2. 设置RKNN输入              │
│  3. 执行推理(rknn_run)        │
│  4. 获取输出张量              │
│  5. 后处理(DFL+NMS)           │
│  6. 恢复原始图像坐标           │
└───────────────────────────────┘
        ↓
yolov8_draw_results() 绘制到LCD
        ↓
显示检测框和类别名称
```

---

## 五、关键技术要点

### 5.1 图像格式要求

YOLOv8 接口要求输入图像必须是 **RGB888 格式**：

| 格式 | 描述 | 是否支持 |
|------|------|----------|
| RGB888 | 每个像素3字节，顺序为R、G、B | **是**（推荐） |
| YUYV | 每个像素2字节，YUV422格式 | 否，需先转换 |
| JPEG | 压缩格式 | 否，需先解码 |

### 5.2 坐标系统

检测结果的坐标是相对于 **原始输入图像** 的：

- (0, 0) 表示图像左上角
- x向右增加，y向下增加
- 边界框由左上角坐标(x, y)和宽高(width, height)定义

### 5.3 置信度阈值

在 [yolov8_detect.cpp](file:///C:/Users/avalon/Desktop/camera/src/yolov8_detect.cpp#L14) 中定义了检测阈值：

```c
#define DETECT_THRESHOLD 0.70f  // 70%置信度
```

只有置信度 >= 70% 的目标才会被绘制到屏幕上，但所有检测结果都会输出到终端。

### 5.4 支持的类别

默认使用 COCO 数据集的80个类别，常见类别如下：

| class_id | 类别名称 | 颜色 |
|----------|----------|------|
| 0 | person（人） | 绿色 |
| 2 | car（汽车） | 蓝色 |
| 5 | bus（公交车） | 黄色 |
| 7 | truck（卡车） | 紫色 |
| 15 | cat（猫） | 红色 |
| 16 | dog（狗） | 红色 |

---

## 六、常见问题与解决方案

### Q1：模型初始化失败怎么办？

**原因分析**：
1. 模型文件路径错误
2. 模型文件损坏
3. RKNN驱动未正确加载

**解决方案**：
```c
// 检查文件是否存在
FILE *fp = fopen("./model/yolov8.rknn", "r");
if (fp == NULL) {
    printf("模型文件不存在！\n");
} else {
    fclose(fp);
}
```

### Q2：检测结果为空？

**原因分析**：
1. 图像格式不正确（不是RGB888）
2. 图像数据为空或损坏
3. 置信度阈值设置过高

**解决方案**：
1. 确保使用 `yuyv_to_rgb888()` 将摄像头数据转换为RGB格式
2. 降低检测阈值（修改 `DETECT_THRESHOLD`）
3. 在终端查看所有检测结果（包括低置信度）

### Q3：检测框位置偏移？

**原因分析**：
- 图像缩放时的 letterbox 变换未正确恢复

**解决方案**：
- 框架内部已自动处理坐标恢复，无需手动干预
- 确保输入图像尺寸与摄像头实际分辨率一致

### Q4：如何添加自己的模型？

**步骤**：
1. 使用 Ultralytics YOLO 训练自己的模型
2. 导出为 ONNX 格式：`yolo export model=best.pt format=onnx`
3. 使用 rknn_toolkit 转换为 RKNN 格式
4. 替换 `model/yolov8.rknn` 和对应的标签文件

---

## 七、总结

调用 YOLOv8 接口实现视觉识别非常简单，只需记住三个核心步骤：

1. **初始化**：`yolov8_init()` 获取模型句柄
2. **检测**：`yolov8_detect_rgb()` 或 `yolov8_detect_and_draw()` 执行检测
3. **释放**：`yolov8_release()` 释放资源

你可以参考其中的代码结构，在自己的项目中轻松集成 YOLOv8 视觉识别功能。
