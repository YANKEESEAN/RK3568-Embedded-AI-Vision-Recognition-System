#ifndef __YOLOV8_WRAPPER_H_
#define __YOLOV8_WRAPPER_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 检测结果结构体
typedef struct {
    int x;              // 边界框左上角x坐标
    int y;              // 边界框左上角y坐标
    int width;          // 边界框宽度
    int height;         // 边界框高度
    float confidence;   // 置信度
    int class_id;       // 类别ID
    char class_name[64]; // 类别名称
} yolov8_detect_result_t;

// 检测结果列表
typedef struct {
    int count;          // 检测到的目标数量
    yolov8_detect_result_t results[128]; // 最多128个检测结果
} yolov8_result_list_t;

// YOLOv8上下文句柄
typedef void* yolov8_context_t;

/**
 * @brief 初始化YOLOv8模型
 * @param model_path 模型文件路径(.rknn)
 * @param labels_path 标签文件路径
 * @return 成功返回上下文句柄，失败返回NULL
 */
yolov8_context_t yolov8_init(const char* model_path, const char* labels_path);

/**
 * @brief 释放YOLOv8模型资源
 * @param ctx YOLOv8上下文句柄
 */
void yolov8_release(yolov8_context_t ctx);

/**
 * @brief 对RGB888图像进行目标检测
 * @param ctx YOLOv8上下文句柄
 * @param rgb_data RGB888格式的图像数据
 * @param width 图像宽度
 * @param height 图像高度
 * @param results 检测结果输出
 * @return 成功返回0，失败返回-1
 */
int yolov8_detect_rgb(yolov8_context_t ctx, 
                      const unsigned char* rgb_data, 
                      int width, 
                      int height,
                      yolov8_result_list_t* results);

/**
 * @brief 对JPEG图像进行目标检测
 * @param ctx YOLOv8上下文句柄
 * @param jpeg_data JPEG格式的图像数据
 * @param jpeg_size JPEG数据大小
 * @param results 检测结果输出
 * @return 成功返回0，失败返回-1
 */
int yolov8_detect_jpeg(yolov8_context_t ctx,
                       const unsigned char* jpeg_data,
                       int jpeg_size,
                       yolov8_result_list_t* results);

/**
 * @brief 获取类别名称
 * @param class_id 类别ID
 * @return 类别名称字符串
 */
const char* yolov8_get_class_name(int class_id);

#ifdef __cplusplus
}
#endif

#endif // __YOLOV8_WRAPPER_H_