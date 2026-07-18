#ifndef __YOLOV8_DETECT_H_
#define __YOLOV8_DETECT_H_

#include "yolov8_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 绘制检测结果到LCD屏幕
 * @param results 检测结果列表
 */
void yolov8_draw_results(yolov8_result_list_t *results);

/**
 * @brief 执行YOLOv8检测并绘制结果
 * @param ctx YOLOv8上下文句柄
 * @param rgb_data RGB888格式的图像数据
 * @param width 图像宽度
 * @param height 图像高度
 * @return 成功返回0，失败返回-1
 */
int yolov8_detect_and_draw(yolov8_context_t ctx, unsigned char *rgb_data, int width, int height);

#ifdef __cplusplus
}
#endif

#endif // __YOLOV8_DETECT_H_
