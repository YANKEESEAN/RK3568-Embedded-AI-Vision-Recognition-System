#include "yolov8_detect.h"
#include "yolov8_wrapper.h"
#include <stdio.h>
#include <string.h>

// LCD绘制函数声明（使用extern "C"避免名称修饰）
extern "C"
{
    void lcd_draw_rect(int x, int y, int w, int h, int color);
    void lcd_draw_text(int x, int y, const char *text, int color);
}

// 检测阈值配置
#define DETECT_THRESHOLD 0.70f // 70%置信度阈值

// 绘制检测结果到LCD屏幕
void yolov8_draw_results(yolov8_result_list_t *results)
{
    if (results == NULL || results->count == 0)
    {
        return;
    }

    for (int i = 0; i < results->count; i++)
    {
        yolov8_detect_result_t *r = &results->results[i];

        // 打印检测结果
        printf("  [%d] %s @ (%d, %d, %d, %d) %.2f%%\n",
               i, r->class_name, r->x, r->y, r->x + r->width, r->y + r->height,
               r->confidence * 100);

        // 只有置信度达到阈值以上才绘制边框和类别名称
        if (r->confidence >= DETECT_THRESHOLD)
        {
            // 根据类别选择颜色
            int color = 0xFF0000; // 默认红色
            if (r->class_id == 0)
                color = 0x00FF00; // person - 绿色
            else if (r->class_id == 2)
                color = 0x0000FF; // car - 蓝色
            else if (r->class_id == 5)
                color = 0xFFFF00; // bus - 黄色
            else if (r->class_id == 7)
                color = 0xFF00FF; // truck - 紫色

            // 绘制检测框
            lcd_draw_rect(r->x, r->y, r->width, r->height, color);

            // 绘制类别名称和置信度（使用main.c中的lcd_draw_text函数）
            char text[128];
            snprintf(text, sizeof(text), "%s %.1f%%", r->class_name, r->confidence * 100);
            lcd_draw_text(r->x, r->y, text, color);

            printf("  -> 置信度达标(>=%.0f%%)，已绘制到屏幕\n", DETECT_THRESHOLD * 100);
        }
        else
        {
            printf("  -> 置信度不足(<%.0f%%)，跳过绘制\n", DETECT_THRESHOLD * 100);
        }
    }
}

// 执行YOLOv8检测并绘制结果
int yolov8_detect_and_draw(yolov8_context_t ctx, unsigned char *rgb_data, int width, int height)
{
    if (ctx == NULL || rgb_data == NULL)
    {
        printf("YOLOv8 上下文或图像数据为空！\n");
        return -1;
    }

    yolov8_result_list_t results;
    int ret = yolov8_detect_rgb(ctx, rgb_data, width, height, &results);

    if (ret == 0)
    {
        if (results.count > 0)
        {
            printf("实时检测到 %d 个目标:\n", results.count);
            yolov8_draw_results(&results);
        }
        else
        {
            printf("未检测到任何目标\n");
        }
    }
    else
    {
        printf("检测失败！ret=%d\n", ret);
    }

    return ret;
}