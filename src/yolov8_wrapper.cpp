// YOLOv8 C++ Wrapper for C interface
// 提供C语言接口调用YOLOv8 RKNN推理

#include "yolov8_wrapper.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <vector>
#include <string>

// RKNN API头文件
#include "rknn_api.h"

#define OBJ_CLASS_NUM 80
#define NMS_THRESH 0.45f
#define BOX_THRESH 0.25f
#define DFL_LEN 16

// 内部结构体定义
typedef struct
{
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;
    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;
    char *labels[OBJ_CLASS_NUM];
} yolov8_context_internal_t;

// letterbox结构体
typedef struct
{
    float x_pad;
    float y_pad;
    float scale;
} letterbox_t;

// 内置 COCO 类别名（定义在文件末尾），标签文件加载失败时用作兜底
const char *yolov8_get_class_name(int class_id);

// 辅助函数声明
static int load_model_data(const char *model_path, char **model_data, int *model_size);
static int load_labels(const char *labels_path, char *labels[], int max_count);
static void dump_tensor_attr(rknn_tensor_attr *attr);
static const char *get_type_string(rknn_tensor_type type);
static const char *get_format_string(rknn_tensor_format fmt);
static const char *get_qnt_type_string(rknn_tensor_qnt_type qnt_type);

// 图像预处理
static int resize_image_with_letterbox(const unsigned char *src, int src_w, int src_h,
                                       unsigned char *dst, int dst_w, int dst_h,
                                       letterbox_t *letterbox);

// 后处理函数
static int post_process_i8(int8_t *box_tensor, int32_t box_zp, float box_scale,
                           int8_t *score_tensor, int32_t score_zp, float score_scale,
                           int grid_h, int grid_w, int stride,
                           std::vector<float> &boxes,
                           std::vector<float> &objProbs,
                           std::vector<int> &classId,
                           float threshold);

static int post_process_u8(uint8_t *box_tensor, int32_t box_zp, float box_scale,
                           uint8_t *score_tensor, int32_t score_zp, float score_scale,
                           int grid_h, int grid_w, int stride,
                           std::vector<float> &boxes,
                           std::vector<float> &objProbs,
                           std::vector<int> &classId,
                           float threshold);

static int post_process_fp32(float *box_tensor, float *score_tensor,
                             int grid_h, int grid_w, int stride,
                             std::vector<float> &boxes,
                             std::vector<float> &objProbs,
                             std::vector<int> &classId,
                             float threshold);

static void compute_dfl(float *tensor, int dfl_len, float *box);
static int nms(std::vector<float> &boxes, std::vector<float> &scores,
               std::vector<int> &classIds, float nms_threshold);
static float calculate_overlap(float xmin0, float ymin0, float xmax0, float ymax0,
                               float xmin1, float ymin1, float xmax1, float ymax1);

// 辅助函数实现
static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, "
           "fmt=%s, type=%s, qnt_type=%s, zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims,
           attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size,
           get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static int load_model_data(const char *model_path, char **model_data, int *model_size)
{
    FILE *fp = fopen(model_path, "rb");
    if (!fp)
    {
        printf("无法打开模型文件: %s\n", model_path);
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    *model_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    *model_data = (char *)malloc(*model_size);
    if (!*model_data)
    {
        fclose(fp);
        return -1;
    }

    fread(*model_data, 1, *model_size, fp);
    fclose(fp);

    return 0;
}

static int load_labels(const char *labels_path, char *labels[], int max_count)
{
    FILE *fp = fopen(labels_path, "r");
    if (!fp)
    {
        printf("无法打开标签文件: %s\n", labels_path);
        return -1;
    }

    char line[256];
    int count = 0;

    while (fgets(line, sizeof(line), fp) && count < max_count)
    {
        // 去除换行符
        int len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
        {
            line[--len] = '\0';
        }
        labels[count] = strdup(line);
        count++;
    }

    fclose(fp);
    return count;
}

// 图像预处理 - 双线性插值缩放 + letterbox
static int resize_image_with_letterbox(const unsigned char *src, int src_w, int src_h,
                                       unsigned char *dst, int dst_w, int dst_h,
                                       letterbox_t *letterbox)
{
    // 计算缩放比例
    float scale = (dst_w < dst_h) ? (float)dst_w / src_w : (float)dst_h / src_h;
    int new_w = (int)(src_w * scale);
    int new_h = (int)(src_h * scale);

    // 计算padding
    letterbox->x_pad = (dst_w - new_w) / 2.0f;
    letterbox->y_pad = (dst_h - new_h) / 2.0f;
    letterbox->scale = scale;

    // 填充背景色 (114, 114, 114)
    for (int i = 0; i < dst_h; i++)
    {
        for (int j = 0; j < dst_w; j++)
        {
            int idx = (i * dst_w + j) * 3;
            dst[idx] = 114;
            dst[idx + 1] = 114;
            dst[idx + 2] = 114;
        }
    }

    // 双线性插值缩放
    for (int y = 0; y < new_h; y++)
    {
        for (int x = 0; x < new_w; x++)
        {
            float src_x = x / scale;
            float src_y = y / scale;

            int x0 = (int)src_x;
            int y0 = (int)src_y;
            int x1 = x0 + 1 < src_w ? x0 + 1 : x0;
            int y1 = y0 + 1 < src_h ? y0 + 1 : y0;

            float fx = src_x - x0;
            float fy = src_y - y0;

            int dst_x = (int)(x + letterbox->x_pad);
            int dst_y = (int)(y + letterbox->y_pad);

            if (dst_x >= 0 && dst_x < dst_w && dst_y >= 0 && dst_y < dst_h)
            {
                int dst_idx = (dst_y * dst_w + dst_x) * 3;

                for (int c = 0; c < 3; c++)
                {
                    int src_idx00 = (y0 * src_w + x0) * 3 + c;
                    int src_idx01 = (y0 * src_w + x1) * 3 + c;
                    int src_idx10 = (y1 * src_w + x0) * 3 + c;
                    int src_idx11 = (y1 * src_w + x1) * 3 + c;

                    float v00 = src[src_idx00];
                    float v01 = src[src_idx01];
                    float v10 = src[src_idx10];
                    float v11 = src[src_idx11];

                    float v0 = v00 * (1 - fx) + v01 * fx;
                    float v1 = v10 * (1 - fx) + v11 * fx;
                    float v = v0 * (1 - fy) + v1 * fy;

                    dst[dst_idx + c] = (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v));
                }
            }
        }
    }

    return 0;
}

// DFL计算
static void compute_dfl(float *tensor, int dfl_len, float *box)
{
    for (int b = 0; b < 4; b++)
    {
        float exp_sum = 0;
        float acc_sum = 0;

        for (int i = 0; i < dfl_len; i++)
        {
            exp_sum += expf(tensor[i + b * dfl_len]);
        }

        for (int i = 0; i < dfl_len; i++)
        {
            acc_sum += (expf(tensor[i + b * dfl_len]) / exp_sum) * i;
        }
        box[b] = acc_sum;
    }
}

// 计算IoU
static float calculate_overlap(float xmin0, float ymin0, float xmax0, float ymax0,
                               float xmin1, float ymin1, float xmax1, float ymax1)
{
    float w = fmaxf(0.0f, fminf(xmax0, xmax1) - fmaxf(xmin0, xmin1));
    float h = fmaxf(0.0f, fminf(ymax0, ymax1) - fmaxf(ymin0, ymin1));
    float i = w * h;
    float u = (xmax0 - xmin0) * (ymax0 - ymin0) + (xmax1 - xmin1) * (ymax1 - ymin1) - i;
    return u <= 0.0f ? 0.0f : (i / u);
}

// NMS
static int nms(std::vector<float> &boxes, std::vector<float> &scores,
               std::vector<int> &classIds, float nms_threshold)
{
    int count = boxes.size() / 4;
    std::vector<int> order(count);
    for (int i = 0; i < count; i++)
        order[i] = i;

    // 按分数排序
    for (int i = 0; i < count - 1; i++)
    {
        for (int j = i + 1; j < count; j++)
        {
            if (scores[order[i]] < scores[order[j]])
            {
                int tmp = order[i];
                order[i] = order[j];
                order[j] = tmp;
            }
        }
    }

    std::vector<bool> suppressed(count, false);

    for (int i = 0; i < count; i++)
    {
        if (suppressed[order[i]])
            continue;

        for (int j = i + 1; j < count; j++)
        {
            if (suppressed[order[j]])
                continue;

            int idx_i = order[i] * 4;
            int idx_j = order[j] * 4;

            float iou = calculate_overlap(
                boxes[idx_i], boxes[idx_i + 1],
                boxes[idx_i] + boxes[idx_i + 2], boxes[idx_i + 1] + boxes[idx_i + 3],
                boxes[idx_j], boxes[idx_j + 1],
                boxes[idx_j] + boxes[idx_j + 2], boxes[idx_j + 1] + boxes[idx_j + 3]);

            if (iou > nms_threshold && classIds[order[i]] == classIds[order[j]])
            {
                suppressed[order[j]] = true;
            }
        }
    }

    // 移除被抑制的结果
    std::vector<float> new_boxes;
    std::vector<float> new_scores;
    std::vector<int> new_classIds;

    for (int i = 0; i < count; i++)
    {
        if (!suppressed[order[i]])
        {
            int idx = order[i] * 4;
            new_boxes.push_back(boxes[idx]);
            new_boxes.push_back(boxes[idx + 1]);
            new_boxes.push_back(boxes[idx + 2]);
            new_boxes.push_back(boxes[idx + 3]);
            new_scores.push_back(scores[order[i]]);
            new_classIds.push_back(classIds[order[i]]);
        }
    }

    boxes = new_boxes;
    scores = new_scores;
    classIds = new_classIds;

    return scores.size();
}

// INT8后处理
static int post_process_i8(int8_t *box_tensor, int32_t box_zp, float box_scale,
                           int8_t *score_tensor, int32_t score_zp, float score_scale,
                           int grid_h, int grid_w, int stride,
                           std::vector<float> &boxes,
                           std::vector<float> &objProbs,
                           std::vector<int> &classId,
                           float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t score_thres_i8 = (int8_t)((threshold / score_scale) + score_zp);
    score_thres_i8 = score_thres_i8 < -128 ? -128 : (score_thres_i8 > 127 ? 127 : score_thres_i8);

    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            int offset = i * grid_w + j;
            int max_class_id = -1;
            int8_t max_score = -score_zp;

            for (int c = 0; c < OBJ_CLASS_NUM; c++)
            {
                int8_t score = score_tensor[offset + c * grid_len];
                if (score > score_thres_i8 && score > max_score)
                {
                    max_score = score;
                    max_class_id = c;
                }
            }

            if (max_score > score_thres_i8)
            {
                float box[4];
                float before_dfl[DFL_LEN * 4];

                for (int k = 0; k < DFL_LEN * 4; k++)
                {
                    before_dfl[k] = ((float)box_tensor[offset + k * grid_len] - box_zp) * box_scale;
                }
                compute_dfl(before_dfl, DFL_LEN, box);

                float x1 = (-box[0] + j + 0.5f) * stride;
                float y1 = (-box[1] + i + 0.5f) * stride;
                float x2 = (box[2] + j + 0.5f) * stride;
                float y2 = (box[3] + i + 0.5f) * stride;
                float w = x2 - x1;
                float h = y2 - y1;

                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);
                objProbs.push_back(((float)max_score - score_zp) * score_scale);
                classId.push_back(max_class_id);
                validCount++;
            }
        }
    }
    return validCount;
}

// UINT8后处理
static int post_process_u8(uint8_t *box_tensor, int32_t box_zp, float box_scale,
                           uint8_t *score_tensor, int32_t score_zp, float score_scale,
                           int grid_h, int grid_w, int stride,
                           std::vector<float> &boxes,
                           std::vector<float> &objProbs,
                           std::vector<int> &classId,
                           float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    uint8_t score_thres_u8 = (uint8_t)((threshold / score_scale) + score_zp);
    score_thres_u8 = score_thres_u8 > 255 ? 255 : score_thres_u8;

    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            int offset = i * grid_w + j;
            int max_class_id = -1;
            uint8_t max_score = 0;

            for (int c = 0; c < OBJ_CLASS_NUM; c++)
            {
                uint8_t score = score_tensor[offset + c * grid_len];
                if (score > score_thres_u8 && score > max_score)
                {
                    max_score = score;
                    max_class_id = c;
                }
            }

            if (max_score > score_thres_u8)
            {
                float box[4];
                float before_dfl[DFL_LEN * 4];

                for (int k = 0; k < DFL_LEN * 4; k++)
                {
                    before_dfl[k] = ((float)box_tensor[offset + k * grid_len] - box_zp) * box_scale;
                }
                compute_dfl(before_dfl, DFL_LEN, box);

                float x1 = (-box[0] + j + 0.5f) * stride;
                float y1 = (-box[1] + i + 0.5f) * stride;
                float x2 = (box[2] + j + 0.5f) * stride;
                float y2 = (box[3] + i + 0.5f) * stride;
                float w = x2 - x1;
                float h = y2 - y1;

                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);
                objProbs.push_back(((float)max_score - score_zp) * score_scale);
                classId.push_back(max_class_id);
                validCount++;
            }
        }
    }
    return validCount;
}

// FP32后处理
static int post_process_fp32(float *box_tensor, float *score_tensor,
                             int grid_h, int grid_w, int stride,
                             std::vector<float> &boxes,
                             std::vector<float> &objProbs,
                             std::vector<int> &classId,
                             float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;

    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            int offset = i * grid_w + j;
            int max_class_id = -1;
            float max_score = 0;

            for (int c = 0; c < OBJ_CLASS_NUM; c++)
            {
                float score = score_tensor[offset + c * grid_len];
                if (score > threshold && score > max_score)
                {
                    max_score = score;
                    max_class_id = c;
                }
            }

            if (max_score > threshold)
            {
                float box[4];
                float before_dfl[DFL_LEN * 4];

                for (int k = 0; k < DFL_LEN * 4; k++)
                {
                    before_dfl[k] = box_tensor[offset + k * grid_len];
                }
                compute_dfl(before_dfl, DFL_LEN, box);

                float x1 = (-box[0] + j + 0.5f) * stride;
                float y1 = (-box[1] + i + 0.5f) * stride;
                float x2 = (box[2] + j + 0.5f) * stride;
                float y2 = (box[3] + i + 0.5f) * stride;
                float w = x2 - x1;
                float h = y2 - y1;

                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);
                objProbs.push_back(max_score);
                classId.push_back(max_class_id);
                validCount++;
            }
        }
    }
    return validCount;
}

// ============ C接口实现 ============

yolov8_context_t yolov8_init(const char *model_path, const char *labels_path)
{
    int ret;

    yolov8_context_internal_t *ctx = (yolov8_context_internal_t *)calloc(1, sizeof(yolov8_context_internal_t));
    if (!ctx)
    {
        printf("上下文内存分配失败\n");
        return NULL;
    }

    // 加载模型
    char *model_data = NULL;
    int model_size = 0;
    ret = load_model_data(model_path, &model_data, &model_size);
    if (ret != 0)
    {
        free(ctx);
        return NULL;
    }

    // 初始化RKNN
    ret = rknn_init(&ctx->rknn_ctx, model_data, (uint32_t)model_size, 0, NULL);
    free(model_data);

    if (ret < 0)
    {
        printf("rknn_init 初始化失败！ret=%d\n", ret);
        free(ctx);
        return NULL;
    }

    // 获取输入输出数量
    ret = rknn_query(ctx->rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &ctx->io_num, sizeof(ctx->io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query 查询失败！ret=%d\n", ret);
        rknn_destroy(ctx->rknn_ctx);
        free(ctx);
        return NULL;
    }

    printf("模型输入数量: %d, 输出数量: %d\n", ctx->io_num.n_input, ctx->io_num.n_output);

    // 获取输入属性
    printf("输入张量信息:\n");
    ctx->input_attrs = (rknn_tensor_attr *)malloc(ctx->io_num.n_input * sizeof(rknn_tensor_attr));
    memset(ctx->input_attrs, 0, ctx->io_num.n_input * sizeof(rknn_tensor_attr));

    for (uint32_t i = 0; i < ctx->io_num.n_input; i++)
    {
        ctx->input_attrs[i].index = i;
        ret = rknn_query(ctx->rknn_ctx, RKNN_QUERY_INPUT_ATTR, &ctx->input_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query 输入属性查询失败！ret=%d\n", ret);
        }
        dump_tensor_attr(&ctx->input_attrs[i]);
    }

    // 获取输出属性
    printf("输出张量信息:\n");
    ctx->output_attrs = (rknn_tensor_attr *)malloc(ctx->io_num.n_output * sizeof(rknn_tensor_attr));
    memset(ctx->output_attrs, 0, ctx->io_num.n_output * sizeof(rknn_tensor_attr));

    for (uint32_t i = 0; i < ctx->io_num.n_output; i++)
    {
        ctx->output_attrs[i].index = i;
        ret = rknn_query(ctx->rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &ctx->output_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query 输出属性查询失败！ret=%d\n", ret);
        }
        dump_tensor_attr(&ctx->output_attrs[i]);
    }

    // 判断是否量化模型
    if (ctx->output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
        ctx->output_attrs[0].type == RKNN_TENSOR_INT8)
    {
        ctx->is_quant = true;
    }
    else if (ctx->output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
             ctx->output_attrs[0].type == RKNN_TENSOR_UINT8)
    {
        ctx->is_quant = true;
    }
    else
    {
        ctx->is_quant = false;
    }

    // 获取模型输入尺寸
    if (ctx->input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        ctx->model_channel = ctx->input_attrs[0].dims[1];
        ctx->model_height = ctx->input_attrs[0].dims[2];
        ctx->model_width = ctx->input_attrs[0].dims[3];
    }
    else
    {
        ctx->model_height = ctx->input_attrs[0].dims[1];
        ctx->model_width = ctx->input_attrs[0].dims[2];
        ctx->model_channel = ctx->input_attrs[0].dims[3];
    }

    printf("模型输入尺寸: 高=%d, 宽=%d, 通道=%d\n",
           ctx->model_height, ctx->model_width, ctx->model_channel);

    // 加载标签
    if (labels_path)
    {
        int label_count = load_labels(labels_path, ctx->labels, OBJ_CLASS_NUM);
        printf("已加载 %d 个标签\n", label_count);
    }

    return ctx;
}

void yolov8_release(yolov8_context_t handle)
{
    yolov8_context_internal_t *ctx = (yolov8_context_internal_t *)handle;
    if (!ctx)
        return;

    if (ctx->input_attrs)
    {
        free(ctx->input_attrs);
    }
    if (ctx->output_attrs)
    {
        free(ctx->output_attrs);
    }
    if (ctx->rknn_ctx)
    {
        rknn_destroy(ctx->rknn_ctx);
    }

    // 释放标签
    for (int i = 0; i < OBJ_CLASS_NUM; i++)
    {
        if (ctx->labels[i])
        {
            free(ctx->labels[i]);
        }
    }

    free(ctx);
}

int yolov8_detect_rgb(yolov8_context_t handle,
                      const unsigned char *rgb_data,
                      int width,
                      int height,
                      yolov8_result_list_t *results)
{
    yolov8_context_internal_t *ctx = (yolov8_context_internal_t *)handle;
    if (!ctx || !rgb_data || !results)
    {
        return -1;
    }

    memset(results, 0, sizeof(yolov8_result_list_t));

    // 预处理 - 缩放图像到模型输入尺寸
    unsigned char *input_data = (unsigned char *)malloc(ctx->model_width * ctx->model_height * 3);
    if (!input_data)
    {
        printf("输入缓冲区内存分配失败\n");
        return -1;
    }

    letterbox_t letterbox;
    resize_image_with_letterbox(rgb_data, width, height, input_data,
                                ctx->model_width, ctx->model_height, &letterbox);

    // 设置输入
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = ctx->model_width * ctx->model_height * ctx->model_channel;
    inputs[0].buf = input_data;

    int ret = rknn_inputs_set(ctx->rknn_ctx, ctx->io_num.n_input, inputs);
    if (ret < 0)
    {
        printf("rknn_inputs_set 设置输入失败！ret=%d\n", ret);
        free(input_data);
        return -1;
    }

    // 推理
    ret = rknn_run(ctx->rknn_ctx, NULL);
    if (ret < 0)
    {
        printf("rknn_run 推理失败！ret=%d\n", ret);
        free(input_data);
        return -1;
    }

    // 获取输出
    rknn_output outputs[ctx->io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (uint32_t i = 0; i < ctx->io_num.n_output; i++)
    {
        outputs[i].index = i;
        outputs[i].want_float = (!ctx->is_quant);
    }

    ret = rknn_outputs_get(ctx->rknn_ctx, ctx->io_num.n_output, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get 获取输出失败！ret=%d\n", ret);
        free(input_data);
        return -1;
    }

    // 后处理
    std::vector<float> boxes;
    std::vector<float> objProbs;
    std::vector<int> classIds;

    int model_in_h = ctx->model_height;

    // YOLOv8有3个输出分支，每个分支有2个输出（box和score）
    int output_per_branch = ctx->io_num.n_output / 3;

    for (int i = 0; i < 3; i++)
    {
        int box_idx = i * output_per_branch;
        int score_idx = i * output_per_branch + 1;

        int grid_h, grid_w, stride;

        // 根据输出格式确定grid尺寸
        if (ctx->output_attrs[box_idx].n_dims == 4)
        {
            // NCHW格式
            grid_h = ctx->output_attrs[box_idx].dims[2];
            grid_w = ctx->output_attrs[box_idx].dims[3];
        }
        else
        {
            grid_h = ctx->output_attrs[box_idx].dims[1];
            grid_w = ctx->output_attrs[box_idx].dims[2];
        }
        stride = model_in_h / grid_h;

        if (ctx->is_quant)
        {
            if (ctx->output_attrs[box_idx].type == RKNN_TENSOR_INT8)
            {
                post_process_i8(
                    (int8_t *)outputs[box_idx].buf,
                    ctx->output_attrs[box_idx].zp,
                    ctx->output_attrs[box_idx].scale,
                    (int8_t *)outputs[score_idx].buf,
                    ctx->output_attrs[score_idx].zp,
                    ctx->output_attrs[score_idx].scale,
                    grid_h, grid_w, stride,
                    boxes, objProbs, classIds, BOX_THRESH);
            }
            else
            {
                post_process_u8(
                    (uint8_t *)outputs[box_idx].buf,
                    ctx->output_attrs[box_idx].zp,
                    ctx->output_attrs[box_idx].scale,
                    (uint8_t *)outputs[score_idx].buf,
                    ctx->output_attrs[score_idx].zp,
                    ctx->output_attrs[score_idx].scale,
                    grid_h, grid_w, stride,
                    boxes, objProbs, classIds, BOX_THRESH);
            }
        }
        else
        {
            post_process_fp32(
                (float *)outputs[box_idx].buf,
                (float *)outputs[score_idx].buf,
                grid_h, grid_w, stride,
                boxes, objProbs, classIds, BOX_THRESH);
        }
    }

    // NMS
    int count = nms(boxes, objProbs, classIds, NMS_THRESH);

    // 转换结果，恢复letterbox变换
    for (int i = 0; i < count && i < 128; i++)
    {
        int idx = i * 4;

        // 恢复到原始图像坐标
        float x1 = (boxes[idx] - letterbox.x_pad) / letterbox.scale;
        float y1 = (boxes[idx + 1] - letterbox.y_pad) / letterbox.scale;
        float w = boxes[idx + 2] / letterbox.scale;
        float h = boxes[idx + 3] / letterbox.scale;

        results->results[i].x = (int)x1;
        results->results[i].y = (int)y1;
        results->results[i].width = (int)w;
        results->results[i].height = (int)h;
        results->results[i].confidence = objProbs[i];
        results->results[i].class_id = classIds[i];

        if (ctx->labels[classIds[i]])
        {
            strncpy(results->results[i].class_name, ctx->labels[classIds[i]], 63);
            results->results[i].class_name[63] = '\0';
        }
        else
        {
            // 标签文件缺失或未成功加载时，使用内置的 COCO 类别名兜底，
            // 保证 class_name 始终非空，避免视觉识别界面上不显示任何文字/色块。
            const char *fallback_name = yolov8_get_class_name(classIds[i]);
            strncpy(results->results[i].class_name, fallback_name, 63);
            results->results[i].class_name[63] = '\0';
        }
    }
    results->count = count < 128 ? count : 128;

    // 释放输出
    rknn_outputs_release(ctx->rknn_ctx, ctx->io_num.n_output, outputs);

    free(input_data);

    return 0;
}

int yolov8_detect_jpeg(yolov8_context_t ctx,
                       const unsigned char *jpeg_data,
                       int jpeg_size,
                       yolov8_result_list_t *results)
{
    // JPEG解码需要libjpeg，这里暂时返回错误
    // 可以在main.c中先解码为RGB再调用yolov8_detect_rgb
    printf("yolov8_detect_jpeg 尚未实现，请改用 yolov8_detect_rgb\n");
    return -1;
}

const char *yolov8_get_class_name(int class_id)
{
    static const char *coco_names[OBJ_CLASS_NUM] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"};

    if (class_id >= 0 && class_id < OBJ_CLASS_NUM)
    {
        return coco_names[class_id];
    }
    return "unknown";
}