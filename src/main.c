#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <linux/fb.h>
#include <linux/input.h>
#include <setjmp.h>
#include <time.h>
#include <dirent.h>
#include <stdbool.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include "jpeglib.h"
#include <png.h>
#include "camera_capture_yuyv.h"

/* ===================== YOLOv8 视觉识别相关 ===================== */
#include "yolov8_wrapper.h"
#include "yolov8_detect.h"

/* 模型/标签文件路径，请根据实际部署路径修改 */
#define YOLO_MODEL_PATH  "./model/yolov8.rknn"
#define YOLO_LABELS_PATH "./model/coco_80_labels_list.txt"

/* 说明：该 YOLOv8 模型基于 COCO 80 类训练，能够识别画面中的“人”(person)、
 * “车”等通用物体类别，属于通用目标检测，并不具备“认出是谁”的人脸身份识别能力。
 * 这里把识别到的 person 目标用绿色框标出，作为“视觉识别 -> 人物检测”演示。
 * 如果需要真正按身份区分人脸（人脸识别 1:N 比对），需要额外接入
 * 人脸检测 + 人脸特征提取(embedding) + 底库比对 的独立模型与流程，
 * 当前 YOLOv8-COCO 模型无法满足该需求。
 */

/* 检测置信度阈值 */
#define VISION_CONF_THRESHOLD 0.5f
/* 每隔多少帧做一次推理，避免每帧都跑模型导致界面卡顿 */
#define VISION_DETECT_INTERVAL 3

static int *double_buffer = NULL;

/* ------------------- 摄像头相关 ------------------- */
int snap_flag = 0;   // 抓拍标志
int camera_flag = 0; // 摄像头运行标志：1=运行中，0=未运行
int record_flag = 0; // 录制标志：1=正在录制
static struct buffer camera_jpeg_buf;
static pthread_t camera_tid;
static int camera_thread_running = 0;
static pthread_mutex_t camera_lock = PTHREAD_MUTEX_INITIALIZER;

#define MAX_PATH_LEN 512
static char album_dir[MAX_PATH_LEN] = "./photos";
static char last_capture_path[MAX_PATH_LEN] = {0};
static int has_last_capture = 0;

static FILE *record_fp = NULL;
static char record_path[MAX_PATH_LEN] = {0};

void start_camera(void);
void stop_camera(void);

/* ------------------- 视觉识别（YOLOv8）相关 ------------------- */
int vision_flag = 0;              // 视觉识别运行标志：1=运行中
static pthread_t vision_tid;
static int vision_thread_running = 0;
static pthread_mutex_t vision_lock = PTHREAD_MUTEX_INITIALIZER;
static struct buffer vision_jpeg_buf;
static yolov8_context_t g_yolo_ctx = NULL; // YOLOv8 推理句柄（程序启动时初始化一次）

void start_vision(void);
void stop_vision(void);
void yolo_init(void);
void yolo_release(void);

/* ------------------- LCD 帧缓冲相关 ------------------- */
int lcd_fd;
int *lcd_ptr;
size_t fb_size;
int screen_width;
int screen_height;

/* ------------------- 触摸屏相关 ------------------- */
int ts_fd;

typedef struct
{
    int x;
    int y;
    int start_x;
    int start_y;
    int pressure;
    bool valid;
} TouchPoint;

/* ------------------- 多点触摸支持 ------------------- */
#define MAX_TOUCH_POINTS 2
static TouchPoint touch_points[MAX_TOUCH_POINTS];
static int touch_point_count = 0;

/* ------------------- 图片缩放相关 ------------------- */
static float current_scale = 1.0f;
static float min_scale = 0.5f;
static float max_scale = 5.0f;
static int image_display_x = 0;
static int image_display_y = 0;
static int image_display_w = 0;
static int image_display_h = 0;

#define MAX_IMAGE_COUNT 100
#define MAX_PATH_LEN 512

/* ------------------- 图片安全显示区域（不遮挡图标） ------------------- */
#define IMAGE_SAFE_LEFT 100
#define IMAGE_SAFE_RIGHT 930
#define IMAGE_SAFE_TOP 0
#define IMAGE_SAFE_BOTTOM 600

/* ------------------- 图标缓存 ------------------- */
typedef struct
{
    char path[MAX_PATH_LEN];
    char *buffer;
    int width;
    int height;
    bool loaded;
} IconCache;

#define MAX_ICON_CACHE 30
static IconCache icon_cache[MAX_ICON_CACHE];
static int icon_cache_count = 0;

/* ------------------- 图标区域定义 ------------------- */
#define ICON_HOME_CAMERA_X 920
#define ICON_HOME_CAMERA_Y 135
#define ICON_HOME_CAMERA_W 90
#define ICON_HOME_CAMERA_H 90

#define ICON_HOME_ALBUM_X 930
#define ICON_HOME_ALBUM_Y 270
#define ICON_HOME_ALBUM_W 80
#define ICON_HOME_ALBUM_H 80

#define ICON_HOME_VISION_X 930
#define ICON_HOME_VISION_Y 410
#define ICON_HOME_VISION_W 75
#define ICON_HOME_VISION_H 75

#define ICON_CAMERA_CLOSE_X 925
#define ICON_CAMERA_CLOSE_Y 180
#define ICON_CAMERA_CLOSE_W 80
#define ICON_CAMERA_CLOSE_H 80

#define ICON_CAMERA_CAPTURE_X 925
#define ICON_CAMERA_CAPTURE_Y 300
#define ICON_CAMERA_CAPTURE_W 85
#define ICON_CAMERA_CAPTURE_H 85

#define ICON_CAMERA_RECORD_X 920
#define ICON_CAMERA_RECORD_Y 440
#define ICON_CAMERA_RECORD_W 100
#define ICON_CAMERA_RECORD_H 100

// 视觉识别页面：关闭按钮区域（复用相机关闭图标位置）
#define ICON_VISION_CLOSE_X 925
#define ICON_VISION_CLOSE_Y 180
#define ICON_VISION_CLOSE_W 80
#define ICON_VISION_CLOSE_H 80

// 摄像头页面：最近抓拍缩略图（位于预览画面下方左下角）
#define ICON_CAMERA_THUMB_X 20
#define ICON_CAMERA_THUMB_Y 480
#define ICON_CAMERA_THUMB_W 140
#define ICON_CAMERA_THUMB_H 100

// 缩略图布局 - 改为5列
#define THUMB_COLS 5
#define THUMB_ROWS 3
#define THUMB_WIDTH 170
#define THUMB_HEIGHT 130
#define THUMB_GAP 12
#define THUMB_TOP 40
#define THUMB_LEFT 30

// 相册缩略图可滑动查看区域（超出范围会被裁剪，避免遮挡退出按钮）
#define ALBUM_VIEW_TOP THUMB_TOP
#define ALBUM_VIEW_BOTTOM 500

// 滚动条样式
#define ALBUM_SCROLLBAR_X 985
#define ALBUM_SCROLLBAR_WIDTH 10

#define ICON_ALBUM_BACK_X 830
#define ICON_ALBUM_BACK_Y 40
#define ICON_ALBUM_BACK_W 100
#define ICON_ALBUM_BACK_H 100

#define ICON_ALBUM_EXIT_X 950
#define ICON_ALBUM_EXIT_Y 520
#define ICON_ALBUM_EXIT_W 80
#define ICON_ALBUM_EXIT_H 80

#define ICON_VIEW_LEFT_X 10
#define ICON_VIEW_LEFT_Y 300
#define ICON_VIEW_LEFT_W 40
#define ICON_VIEW_LEFT_H 40

#define ICON_VIEW_RIGHT_X 970
#define ICON_VIEW_RIGHT_Y 300
#define ICON_VIEW_RIGHT_W 40
#define ICON_VIEW_RIGHT_H 40

// 右上角退出图标（全屏查看时）
#define ICON_VIEW_EXIT_X 970
#define ICON_VIEW_EXIT_Y 20
#define ICON_VIEW_EXIT_W 50
#define ICON_VIEW_EXIT_H 50

// 右下角特效切换图标（全屏查看时）
#define ICON_VIEW_EFFECT_X 970
#define ICON_VIEW_EFFECT_Y 510
#define ICON_VIEW_EFFECT_W 50
#define ICON_VIEW_EFFECT_H 50

// 左下角放大和缩小图标
#define ICON_VIEW_ZOOM_IN_X 20
#define ICON_VIEW_ZOOM_IN_Y 400
#define ICON_VIEW_ZOOM_IN_W 50
#define ICON_VIEW_ZOOM_IN_H 50

#define ICON_VIEW_ZOOM_OUT_X 20
#define ICON_VIEW_ZOOM_OUT_Y 500
#define ICON_VIEW_ZOOM_OUT_W 50
#define ICON_VIEW_ZOOM_OUT_H 50

#define ICON_EXIT_X 920
#define ICON_EXIT_Y 50
#define ICON_EXIT_W 90
#define ICON_EXIT_H 90

/* ------------------- 页面状态定义 ------------------- */
typedef enum
{
    PAGE_LOCKSCREEN,
    PAGE_HOME,
    PAGE_CAMERA,
    PAGE_ALBUM,
    PAGE_VIEW,
    PAGE_VISION
} PageState;

static PageState current_page = PAGE_LOCKSCREEN;
void draw_icons_overlay(void);

/* ------------------- 图片资源路径定义 ------------------- */
#define IMAGE_LOCKSCREEN "images/lockscreen.jpg"
#define IMAGE_BACKGROUND "images/background.jpg"
#define IMAGE_ICON_CLOSE "images/close.png"
#define IMAGE_ICON_CAPTURE "images/capture.png"
#define IMAGE_ICON_RECORD "images/record.png"
#define IMAGE_ICON_ALBUM "images/album.jpg"
#define IMAGE_ICON_VISION "images/vision.jpg"
#define IMAGE_ICON_PREV "images/left.png"
#define IMAGE_ICON_NEXT "images/right.png"
#define IMAGE_ICON_EFFECT "images/effect.png"
#define IMAGE_ICON_EXIT "images/exit.png"
#define IMAGE_ICON_BACK "images/back.jpg"
#define IMAGE_ICON_HOME_CAMERA "images/home_camera.png"
#define IMAGE_ICON_HOME_ALBUM "images/home_album.png"
#define IMAGE_ICON_HOME_VISION "images/home_vision.png"
#define IMAGE_ICON_ZOOM_IN "images/zoomin.png"
#define IMAGE_ICON_ZOOM_OUT "images/zoomout.png"
#define IMAGE_ICON_SWIPE_UP "images/swipeup.png"

/* ============================================================ */
/* =============== 目录检索功能 =============== */
/* ============================================================ */

static char image_paths[MAX_IMAGE_COUNT][MAX_PATH_LEN];
static int image_count = 0;
static int current_image_index = 0;
static int current_effect_index = 0;
static int album_scroll_offset = 0; // 相册缩略图垂直滚动偏移（像素）

// 添加静态变量用于缓存首页内容
static char *home_background_buffer = NULL;
static int home_bg_width = 0;
static int home_bg_height = 0;
static bool home_cached = false;

typedef struct
{
    char path[MAX_PATH_LEN];
    char *buffer;
    int width;
    int height;
    bool loaded;
} ThumbnailCache;

static ThumbnailCache thumbnail_cache[MAX_IMAGE_COUNT];

int scan_directory(const char *route, const char *type)
{
    char full_path[MAX_PATH_LEN] = {0};
    int count = 0;

    DIR *dirp = opendir(route);
    if (dirp == NULL)
    {
        printf("无法打开目录: %s\n", route);
        return -1;
    }

    printf("正在扫描目录: %s (查找 %s 文件)...\n", route, type);

    while (1)
    {
        struct dirent *dir_ptr = readdir(dirp);

        if (dir_ptr == NULL)
        {
            break;
        }

        if (!strncmp(dir_ptr->d_name, ".", 1) || !strncmp(dir_ptr->d_name, "..", 2))
        {
            continue;
        }

        snprintf(full_path, sizeof(full_path), "%s/%s", route, dir_ptr->d_name);

        if (dir_ptr->d_type == DT_REG)
        {
            int name_len = strlen(full_path);
            int type_len = strlen(type);

            if (name_len > type_len)
            {
                if (strncasecmp(full_path + name_len - type_len, type, type_len) == 0)
                {
                    strncpy(image_paths[count], full_path, MAX_PATH_LEN - 1);
                    image_paths[count][MAX_PATH_LEN - 1] = '\0';

                    strncpy(thumbnail_cache[count].path, full_path, MAX_PATH_LEN - 1);
                    thumbnail_cache[count].buffer = NULL;
                    thumbnail_cache[count].loaded = false;

                    count++;
                    printf("  找到: %s\n", full_path);
                }
            }
        }
    }

    closedir(dirp);
    image_count = count;
    printf("共找到 %d 个 %s 文件。\n\n", count, type);
    return count;
}

void print_image_list()
{
    printf("=== 图片列表 ===\n");
    for (int i = 0; i < image_count; i++)
    {
        printf("[%d] %s\n", i, image_paths[i]);
    }
    printf("总计: %d 张图片\n", image_count);
}

/* ============================================================ */
/* =============== 触摸屏功能 =============== */
/* ============================================================ */

// 获取单点触摸数据（保留原接口）
TouchPoint get_touch_data(int fd)
{
    struct input_event ev;
    static TouchPoint point = {0};
    bool start_captured = false;

    point.valid = false;
    point.start_x = point.x;
    point.start_y = point.y;

    while (read(fd, &ev, sizeof(ev)) == sizeof(ev))
    {
        switch (ev.type)
        {
        case EV_ABS:
            switch (ev.code)
            {
            case ABS_X:
                point.x = ev.value;
                if (!start_captured)
                {
                    point.start_x = ev.value;
                }
                break;
            case ABS_Y:
                point.y = ev.value;
                if (!start_captured)
                {
                    point.start_y = ev.value;
                    start_captured = true;
                }
                break;
            case ABS_PRESSURE:
                point.pressure = ev.value;
                break;
            }
            break;

        case EV_KEY:
            if (ev.code == BTN_TOUCH && ev.value == 0)
            {
                point.valid = true;
                return point;
            }
            break;
        }
    }

    return (TouchPoint){0};
}

// 获取多点触摸数据（支持两指）
int get_multi_touch_data(int fd, TouchPoint points[], int max_points)
{
    struct input_event ev;
    static TouchPoint tracked_points[10] = {0};
    static int point_ids[10] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    int count = 0;

    while (read(fd, &ev, sizeof(ev)) == sizeof(ev))
    {
        switch (ev.type)
        {
        case EV_ABS:
            switch (ev.code)
            {
            case ABS_MT_TRACKING_ID:
                if (ev.value >= 0 && ev.value < 10)
                {
                    point_ids[ev.value] = ev.value;
                }
                break;
            case ABS_MT_POSITION_X:
                break;
            case ABS_MT_POSITION_Y:
                break;
            case ABS_MT_PRESSURE:
                break;
            }
            break;
        case EV_KEY:
            if (ev.code == BTN_TOUCH && ev.value == 0)
            {
                for (int i = 0; i < 10; i++)
                {
                    point_ids[i] = -1;
                }
                return 0;
            }
            break;
        case EV_SYN:
            break;
        }
    }

    TouchPoint single = get_touch_data(fd);
    if (single.valid)
    {
        points[0] = single;
        return 1;
    }

    return 0;
}

// 获取两指触摸距离
float get_two_finger_distance(TouchPoint p1, TouchPoint p2)
{
    int dx = p1.x - p2.x;
    int dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

bool is_in_region(int x, int y, int region_x, int region_y, int region_w, int region_h)
{
    return (x >= region_x && x <= region_x + region_w &&
            y >= region_y && y <= region_y + region_h);
}

/* ============================================================ */
/* =============== LCD 设备初始化 =============== */
/* ============================================================ */

int dev_init()
{
    lcd_fd = open("/dev/fb0", O_RDWR);
    if (-1 == lcd_fd)
    {
        printf("打开 LCD 设备失败!\n");
        return -1;
    }
    struct fb_var_screeninfo vinfo;
    struct fb_fix_screeninfo finfo;
    ioctl(lcd_fd, FBIOGET_FSCREENINFO, &finfo);
    ioctl(lcd_fd, FBIOGET_VSCREENINFO, &vinfo);

    screen_width = vinfo.xres;
    screen_height = vinfo.yres;
    fb_size = vinfo.xres * vinfo.yres * vinfo.bits_per_pixel / 8;

    lcd_ptr = mmap(NULL, fb_size, PROT_READ | PROT_WRITE, MAP_SHARED, lcd_fd, 0);
    if (lcd_ptr == MAP_FAILED)
    {
        perror("映射帧缓冲失败");
        close(lcd_fd);
        return -1;
    }

    double_buffer = (int *)malloc(fb_size);
    if (double_buffer == NULL)
    {
        printf("分配双缓冲内存失败!\n");
        return -1;
    }
    memset(double_buffer, 0, fb_size);

    ts_fd = open("/dev/input/event6", O_RDONLY);
    if (ts_fd == -1)
    {
        perror("打开触摸设备失败");
        return -1;
    }

    return 0;
}

void dev_uninit()
{
    stop_camera();
    stop_vision();
    yolo_release();

    for (int i = 0; i < image_count; i++)
    {
        if (thumbnail_cache[i].buffer != NULL)
        {
            free(thumbnail_cache[i].buffer);
            thumbnail_cache[i].buffer = NULL;
        }
    }

    // 释放首页背景缓存
    if (home_background_buffer != NULL)
    {
        free(home_background_buffer);
        home_background_buffer = NULL;
        home_cached = false;
    }

    for (int i = 0; i < icon_cache_count; i++)
    {
        if (icon_cache[i].buffer != NULL)
        {
            free(icon_cache[i].buffer);
            icon_cache[i].buffer = NULL;
            icon_cache[i].loaded = false;
        }
    }
    icon_cache_count = 0;

    if (double_buffer != NULL)
    {
        free(double_buffer);
        double_buffer = NULL;
    }
    munmap(lcd_ptr, fb_size);
    close(lcd_fd);
    close(ts_fd);
}

void clear_screen()
{
    memset(lcd_ptr, 0, fb_size);
}

void clear_double_buffer()
{
    memset(double_buffer, 0, fb_size);
}

void flush_double_buffer()
{
    memcpy(lcd_ptr, double_buffer, fb_size);
}

/* ============================================================ */
/* =============== JPEG 解码 =============== */
/* ============================================================ */

struct my_error_mgr
{
    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
};
typedef struct my_error_mgr *my_error_ptr;

METHODDEF(void)
my_error_exit(j_common_ptr cinfo)
{
    my_error_ptr myerr = (my_error_ptr)cinfo->err;
    (*cinfo->err->output_message)(cinfo);
    longjmp(myerr->setjmp_buffer, 1);
}

GLOBAL(int)
read_JPEG_file(char *filename, int *image_width, int *image_height, char **out_buffer)
{
    struct jpeg_decompress_struct cinfo;
    struct my_error_mgr jerr;
    FILE *infile;
    JSAMPARRAY buffer;
    int row_stride;

    if ((infile = fopen(filename, "rb")) == NULL)
    {
        fprintf(stderr, "无法打开 %s\n", filename);
        return 0;
    }

    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit;
    if (setjmp(jerr.setjmp_buffer))
    {
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        return 0;
    }

    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    (void)jpeg_read_header(&cinfo, TRUE);
    (void)jpeg_start_decompress(&cinfo);

    *image_width = cinfo.output_width;
    *image_height = cinfo.output_height;
    row_stride = cinfo.output_width * cinfo.output_components;

    *out_buffer = (char *)malloc(row_stride * cinfo.output_height);
    if (*out_buffer == NULL)
    {
        printf("内存分配失败!\n");
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        return 0;
    }

    char *dst = *out_buffer;
    buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1);

    while (cinfo.output_scanline < cinfo.output_height)
    {
        (void)jpeg_read_scanlines(&cinfo, buffer, 1);
        memcpy(dst, buffer[0], row_stride);
        dst += row_stride;
    }

    (void)jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    return 1;
}

// 从内存缓冲区解码 JPEG（用于摄像头实时帧，无需落盘）
int read_JPEG_mem(unsigned char *data, unsigned long data_size,
                  int *image_width, int *image_height, char **out_buffer)
{
    struct jpeg_decompress_struct cinfo;
    struct my_error_mgr jerr;
    JSAMPARRAY buffer;
    int row_stride;

    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit;
    if (setjmp(jerr.setjmp_buffer))
    {
        jpeg_destroy_decompress(&cinfo);
        return 0;
    }

    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, data, data_size);
    (void)jpeg_read_header(&cinfo, TRUE);
    (void)jpeg_start_decompress(&cinfo);

    *image_width = cinfo.output_width;
    *image_height = cinfo.output_height;
    row_stride = cinfo.output_width * cinfo.output_components;

    *out_buffer = (char *)malloc(row_stride * cinfo.output_height);
    if (*out_buffer == NULL)
    {
        jpeg_destroy_decompress(&cinfo);
        return 0;
    }

    char *dst = *out_buffer;
    buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1);

    while (cinfo.output_scanline < cinfo.output_height)
    {
        (void)jpeg_read_scanlines(&cinfo, buffer, 1);
        memcpy(dst, buffer[0], row_stride);
        dst += row_stride;
    }

    (void)jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    return 1;
}

/* ============================================================ */
/* =============== PNG 解码 =============== */
/* ============================================================ */

int read_PNG_file(const char *filename, int *image_width, int *image_height, char **out_buffer)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        fprintf(stderr, "无法打开 %s\n", filename);
        return 0;
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png)
    {
        fclose(fp);
        return 0;
    }

    png_infop info = png_create_info_struct(png);
    if (!info)
    {
        png_destroy_read_struct(&png, NULL, NULL);
        fclose(fp);
        return 0;
    }

    if (setjmp(png_jmpbuf(png)))
    {
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        return 0;
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    *image_width = png_get_image_width(png, info);
    *image_height = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    if (bit_depth == 16)
    {
        png_set_strip_16(png);
    }
    if (color_type == PNG_COLOR_TYPE_PALETTE)
    {
        png_set_palette_to_rgb(png);
    }
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
    {
        png_set_expand_gray_1_2_4_to_8(png);
    }
    if (png_get_valid(png, info, PNG_INFO_tRNS))
    {
        png_set_tRNS_to_alpha(png);
    }
    if (color_type == PNG_COLOR_TYPE_RGB_ALPHA || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
    {
        png_set_strip_alpha(png);
    }

    png_read_update_info(png, info);

    int row_bytes = png_get_rowbytes(png, info);
    *out_buffer = (char *)malloc(row_bytes * (*image_height));
    if (!*out_buffer)
    {
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        return 0;
    }

    png_bytep *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * (*image_height));
    for (int y = 0; y < *image_height; y++)
    {
        row_pointers[y] = (png_bytep)(*out_buffer + y * row_bytes);
    }

    png_read_image(png, row_pointers);
    free(row_pointers);

    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);
    return 1;
}

/* ============================================================ */
/* =============== 通用图片加载函数 =============== */
/* ============================================================ */

int load_image(const char *filename, int *width, int *height, char **buffer)
{
    const char *ext = strrchr(filename, '.');
    if (ext == NULL)
    {
        printf("无扩展名: %s\n", filename);
        return 0;
    }

    if (strcasecmp(ext, ".jpg") == 0 || strcasecmp(ext, ".jpeg") == 0)
    {
        return read_JPEG_file((char *)filename, width, height, buffer);
    }
    else if (strcasecmp(ext, ".png") == 0)
    {
        return read_PNG_file(filename, width, height, buffer);
    }
    else
    {
        printf("不支持的格式: %s\n", ext);
        return 0;
    }
}

/* ============================================================ */
/* =============== 图片绘制（支持缩放和区域限制） =============== */
/* ============================================================ */

// 绘制缩放后的图片到缓冲区（支持缩放偏移和区域限制）
static void draw_scaled_image_to_buffer_with_offset_limited(char *buffer, int w, int h,
                                                            int draw_w, int draw_h,
                                                            int offset_x, int offset_y,
                                                            float scale,
                                                            int limit_x, int limit_y, int limit_w, int limit_h)
{
    int actual_w = draw_w;
    int actual_h = draw_h;
    int actual_offset_x = offset_x;
    int actual_offset_y = offset_y;

    // 如果scale不为1.0，应用缩放
    if (scale != 1.0f)
    {
        int new_w = (int)(draw_w * scale);
        int new_h = (int)(draw_h * scale);
        int center_x = offset_x + draw_w / 2;
        int center_y = offset_y + draw_h / 2;
        actual_w = new_w;
        actual_h = new_h;
        actual_offset_x = center_x - new_w / 2;
        actual_offset_y = center_y - new_h / 2;
    }

    // 确保不超出屏幕边界
    if (actual_offset_x + actual_w < 0 || actual_offset_x > screen_width ||
        actual_offset_y + actual_h < 0 || actual_offset_y > screen_height)
    {
        return;
    }

    float src_scale_x = (float)w / actual_w;
    float src_scale_y = (float)h / actual_h;

    // 绘制图片到指定区域，但限制在 limit 区域内
    for (int py = actual_offset_y; py < actual_offset_y + actual_h; py++)
    {
        if (py < 0 || py >= screen_height)
            continue;
        // 检查是否在限制区域内
        if (py < limit_y || py >= limit_y + limit_h)
            continue;

        int src_y = (int)((py - actual_offset_y) * src_scale_y);
        if (src_y >= h)
            src_y = h - 1;
        int row_offset = src_y * w * 3;

        for (int px = actual_offset_x; px < actual_offset_x + actual_w; px++)
        {
            if (px < 0 || px >= screen_width)
                continue;
            // 检查是否在限制区域内
            if (px < limit_x || px >= limit_x + limit_w)
                continue;

            int src_x = (int)((px - actual_offset_x) * src_scale_x);
            if (src_x >= w)
                src_x = w - 1;
            int idx = row_offset + src_x * 3;

            int r = buffer[idx + 0];
            int g = buffer[idx + 1];
            int b = buffer[idx + 2];
            int color = (r << 16) | (g << 8) | b;
            double_buffer[py * screen_width + px] = color;
        }
    }
}

// 原始绘制函数（保持兼容）
static void draw_scaled_image_to_buffer(char *buffer, int w, int h, int draw_w, int draw_h,
                                        int offset_x, int offset_y)
{
    draw_scaled_image_to_buffer_with_offset_limited(buffer, w, h, draw_w, draw_h, offset_x, offset_y, 1.0f,
                                                    IMAGE_SAFE_LEFT, IMAGE_SAFE_TOP,
                                                    IMAGE_SAFE_RIGHT - IMAGE_SAFE_LEFT,
                                                    IMAGE_SAFE_BOTTOM - IMAGE_SAFE_TOP);
}

/* ============================================================ */
/* =============== 缩略图加载与缓存 =============== */
/* ============================================================ */

void load_thumbnail(int index)
{
    if (index < 0 || index >= image_count)
        return;
    if (thumbnail_cache[index].loaded)
        return;

    int w, h;
    char *buffer = NULL;

    if (load_image(thumbnail_cache[index].path, &w, &h, &buffer))
    {
        thumbnail_cache[index].buffer = buffer;
        thumbnail_cache[index].width = w;
        thumbnail_cache[index].height = h;
        thumbnail_cache[index].loaded = true;
    }
}

void load_all_thumbnails()
{
    for (int i = 0; i < image_count; i++)
    {
        load_thumbnail(i);
    }
}

void free_thumbnail_cache()
{
    for (int i = 0; i < image_count; i++)
    {
        if (thumbnail_cache[i].buffer != NULL)
        {
            free(thumbnail_cache[i].buffer);
            thumbnail_cache[i].buffer = NULL;
            thumbnail_cache[i].loaded = false;
        }
    }
}

/* ============================================================ */
/* =============== 图片绘制函数（特效只影响图片区域） =============== */
/* ============================================================ */

typedef void (*draw_callback_t)(char *buffer, int w, int h, int display_w, int display_h, int offset_x, int offset_y);

static int v_progress = 0;
static int h_progress = 0;
static int e_progress = 0;
static int blind_progress = 0;

static char *current_buffer = NULL;
static int current_width = 0;
static int current_height = 0;

// 全屏图片显示区域（用于特效）
static int g_display_x = 0;
static int g_display_y = 0;
static int g_display_w = 0;
static int g_display_h = 0;

// 计算图片在屏幕上的显示区域（居中，保持宽高比）
void calculate_display_area(int img_w, int img_h, int *disp_w, int *disp_h, int *off_x, int *off_y)
{
    float scale_x = (float)screen_width / img_w;
    float scale_y = (float)screen_height / img_h;
    float scale = (scale_x < scale_y) ? scale_x : scale_y;

    *disp_w = (int)(img_w * scale);
    *disp_h = (int)(img_h * scale);
    *off_x = (screen_width - *disp_w) / 2;
    *off_y = (screen_height - *disp_h) / 2;
}

void draw_normal_to_buffer(char *buffer, int w, int h, int disp_w, int disp_h, int off_x, int off_y)
{
    draw_scaled_image_to_buffer_with_offset_limited(buffer, w, h, disp_w, disp_h, off_x, off_y,
                                                    current_scale,
                                                    IMAGE_SAFE_LEFT, IMAGE_SAFE_TOP,
                                                    IMAGE_SAFE_RIGHT - IMAGE_SAFE_LEFT,
                                                    IMAGE_SAFE_BOTTOM - IMAGE_SAFE_TOP);
}

void draw_vertical_slide_to_buffer(char *buffer, int w, int h, int disp_w, int disp_h, int off_x, int off_y)
{
    int draw_h = (v_progress * disp_h) / 100;
    if (draw_h > disp_h)
        draw_h = disp_h;
    draw_scaled_image_to_buffer_with_offset_limited(buffer, w, h, disp_w, draw_h, off_x, off_y,
                                                    current_scale,
                                                    IMAGE_SAFE_LEFT, IMAGE_SAFE_TOP,
                                                    IMAGE_SAFE_RIGHT - IMAGE_SAFE_LEFT,
                                                    IMAGE_SAFE_BOTTOM - IMAGE_SAFE_TOP);
}

void draw_horizontal_slide_to_buffer(char *buffer, int w, int h, int disp_w, int disp_h, int off_x, int off_y)
{
    int draw_w = (h_progress * disp_w) / 100;
    if (draw_w > disp_w)
        draw_w = disp_w;
    draw_scaled_image_to_buffer_with_offset_limited(buffer, w, h, draw_w, disp_h, off_x, off_y,
                                                    current_scale,
                                                    IMAGE_SAFE_LEFT, IMAGE_SAFE_TOP,
                                                    IMAGE_SAFE_RIGHT - IMAGE_SAFE_LEFT,
                                                    IMAGE_SAFE_BOTTOM - IMAGE_SAFE_TOP);
}

void draw_center_expand_to_buffer(char *buffer, int w, int h, int disp_w, int disp_h, int off_x, int off_y)
{
    int draw_w = (e_progress * disp_w) / 100;
    int draw_h = (e_progress * disp_h) / 100;
    if (draw_w < 1)
        draw_w = 1;
    if (draw_h < 1)
        draw_h = 1;
    int new_off_x = off_x + (disp_w - draw_w) / 2;
    int new_off_y = off_y + (disp_h - draw_h) / 2;
    draw_scaled_image_to_buffer_with_offset_limited(buffer, w, h, draw_w, draw_h, new_off_x, new_off_y,
                                                    current_scale,
                                                    IMAGE_SAFE_LEFT, IMAGE_SAFE_TOP,
                                                    IMAGE_SAFE_RIGHT - IMAGE_SAFE_LEFT,
                                                    IMAGE_SAFE_BOTTOM - IMAGE_SAFE_TOP);
}

void draw_horizontal_blind_to_buffer(char *buffer, int w, int h, int disp_w, int disp_h, int off_x, int off_y)
{
    int blind_count = 10;
    int blind_height = disp_h / blind_count;
    int reveal_height = (blind_progress * blind_height) / 100;

    draw_scaled_image_to_buffer_with_offset_limited(buffer, w, h, disp_w, disp_h, off_x, off_y,
                                                    current_scale,
                                                    IMAGE_SAFE_LEFT, IMAGE_SAFE_TOP,
                                                    IMAGE_SAFE_RIGHT - IMAGE_SAFE_LEFT,
                                                    IMAGE_SAFE_BOTTOM - IMAGE_SAFE_TOP);

    for (int py = off_y; py < off_y + disp_h; py++)
    {
        if (py < 0 || py >= screen_height)
            continue;
        if (py < IMAGE_SAFE_TOP || py >= IMAGE_SAFE_BOTTOM)
            continue;
        int local_y = py - off_y;
        int blind_index = local_y / blind_height;
        int offset_in_blind = local_y % blind_height;
        if (offset_in_blind >= reveal_height)
        {
            for (int px = off_x; px < off_x + disp_w; px++)
            {
                if (px < 0 || px >= screen_width)
                    continue;
                if (px < IMAGE_SAFE_LEFT || px >= IMAGE_SAFE_RIGHT)
                    continue;
                double_buffer[py * screen_width + px] = 0;
            }
        }
    }
}

void draw_vertical_blind_to_buffer(char *buffer, int w, int h, int disp_w, int disp_h, int off_x, int off_y)
{
    int blind_count = 10;
    int blind_width = disp_w / blind_count;
    int reveal_width = (blind_progress * blind_width) / 100;

    draw_scaled_image_to_buffer_with_offset_limited(buffer, w, h, disp_w, disp_h, off_x, off_y,
                                                    current_scale,
                                                    IMAGE_SAFE_LEFT, IMAGE_SAFE_TOP,
                                                    IMAGE_SAFE_RIGHT - IMAGE_SAFE_LEFT,
                                                    IMAGE_SAFE_BOTTOM - IMAGE_SAFE_TOP);

    for (int py = off_y; py < off_y + disp_h; py++)
    {
        if (py < 0 || py >= screen_height)
            continue;
        if (py < IMAGE_SAFE_TOP || py >= IMAGE_SAFE_BOTTOM)
            continue;
        for (int px = off_x; px < off_x + disp_w; px++)
        {
            if (px < 0 || px >= screen_width)
                continue;
            if (px < IMAGE_SAFE_LEFT || px >= IMAGE_SAFE_RIGHT)
                continue;
            int local_x = px - off_x;
            int blind_index = local_x / blind_width;
            int offset_in_blind = local_x % blind_width;
            if (offset_in_blind >= reveal_width)
            {
                double_buffer[py * screen_width + px] = 0;
            }
        }
    }
}

draw_callback_t effect_functions[] = {
    draw_normal_to_buffer,
    draw_vertical_slide_to_buffer,
    draw_horizontal_slide_to_buffer,
    draw_center_expand_to_buffer,
    draw_horizontal_blind_to_buffer,
    draw_vertical_blind_to_buffer};

const char *effect_names[] = {
    "正常显示",
    "垂直卷帘",
    "水平卷帘",
    "中心扩散",
    "水平百叶窗",
    "垂直百叶窗"};

#define EFFECT_COUNT (sizeof(effect_functions) / sizeof(effect_functions[0]))

int lcd_load_image(const char *pathname)
{
    int image_width, image_height;
    char *out_buffer = NULL;

    if (!load_image(pathname, &image_width, &image_height, &out_buffer))
    {
        printf("解码图片失败: %s\n", pathname);
        return -1;
    }

    if (current_buffer != NULL)
    {
        free(current_buffer);
        current_buffer = NULL;
    }

    current_buffer = out_buffer;
    current_width = image_width;
    current_height = image_height;

    return 0;
}

// 切换图片时重置缩放
void switch_image(int index)
{
    if (index < 0 || index >= image_count)
        return;
    current_image_index = index;
    current_scale = 1.0f;
}

void draw_current_image_to_buffer(draw_callback_t effect_func)
{
    if (current_buffer == NULL)
    {
        printf("未加载图片!\n");
        return;
    }

    calculate_display_area(current_width, current_height, &g_display_w, &g_display_h, &g_display_x, &g_display_y);

    v_progress = 0;
    h_progress = 0;
    e_progress = 0;
    blind_progress = 0;

    int steps = 18;
    int usleep_delay = 1500 * 1000 / steps;

    for (int progress = 10; progress <= 100; progress += 5)
    {
        if (effect_func == draw_vertical_slide_to_buffer)
        {
            v_progress = progress;
        }
        else if (effect_func == draw_horizontal_slide_to_buffer)
        {
            h_progress = progress;
        }
        else if (effect_func == draw_center_expand_to_buffer)
        {
            e_progress = progress;
        }
        else if (effect_func == draw_horizontal_blind_to_buffer ||
                 effect_func == draw_vertical_blind_to_buffer)
        {
            blind_progress = progress;
        }

        clear_double_buffer();
        effect_func(current_buffer, current_width, current_height, g_display_w, g_display_h, g_display_x, g_display_y);
        if (current_page == PAGE_VIEW)
        {
            draw_icons_overlay();
        }
        flush_double_buffer();
        usleep(usleep_delay);
    }

    if (effect_func != draw_normal_to_buffer)
    {
        if (effect_func == draw_vertical_slide_to_buffer)
            v_progress = 100;
        else if (effect_func == draw_horizontal_slide_to_buffer)
            h_progress = 100;
        else if (effect_func == draw_center_expand_to_buffer)
            e_progress = 100;
        else if (effect_func == draw_horizontal_blind_to_buffer ||
                 effect_func == draw_vertical_blind_to_buffer)
            blind_progress = 100;

        clear_double_buffer();
        effect_func(current_buffer, current_width, current_height, g_display_w, g_display_h, g_display_x, g_display_y);
        if (current_page == PAGE_VIEW)
        {
            draw_icons_overlay();
        }
        flush_double_buffer();
    }
}

void show_image_to_buffer(const char *image_path, draw_callback_t effect_func)
{
    if (lcd_load_image(image_path) != 0)
    {
        return;
    }
    draw_current_image_to_buffer(effect_func);
}

void release_current_image()
{
    if (current_buffer != NULL)
    {
        free(current_buffer);
        current_buffer = NULL;
    }
}

/* ============================================================ */
/* =============== 图标缓存与绘制 =============== */
/* ============================================================ */

// 获取或加载图标（带缓存）
char *get_cached_icon(const char *icon_path, int *width, int *height)
{
    for (int i = 0; i < icon_cache_count; i++)
    {
        if (strcmp(icon_cache[i].path, icon_path) == 0)
        {
            if (width)
                *width = icon_cache[i].width;
            if (height)
                *height = icon_cache[i].height;
            return icon_cache[i].buffer;
        }
    }

    char *buffer = NULL;
    int w, h;
    if (!load_image(icon_path, &w, &h, &buffer))
    {
        printf("加载图标失败: %s\n", icon_path);
        return NULL;
    }

    if (icon_cache_count < MAX_ICON_CACHE)
    {
        strncpy(icon_cache[icon_cache_count].path, icon_path, MAX_PATH_LEN - 1);
        icon_cache[icon_cache_count].path[MAX_PATH_LEN - 1] = '\0';
        icon_cache[icon_cache_count].buffer = buffer;
        icon_cache[icon_cache_count].width = w;
        icon_cache[icon_cache_count].height = h;
        icon_cache[icon_cache_count].loaded = true;
        if (width)
            *width = w;
        if (height)
            *height = h;
        icon_cache_count++;
        return buffer;
    }

    return NULL;
}

void draw_icon_to_buffer(const char *icon_path, int x, int y, int width, int height)
{
    int img_w, img_h;
    char *buffer = get_cached_icon(icon_path, &img_w, &img_h);

    if (buffer == NULL)
    {
        printf("加载图标失败: %s\n", icon_path);
        return;
    }

    float scale_x = (float)img_w / width;
    float scale_y = (float)img_h / height;

    for (int py = y; py < y + height; py++)
    {
        if (py < 0 || py >= screen_height)
            continue;
        int src_y = (int)((py - y) * scale_y);
        if (src_y >= img_h)
            src_y = img_h - 1;
        int row_offset = src_y * img_w * 3;

        for (int px = x; px < x + width; px++)
        {
            if (px < 0 || px >= screen_width)
                continue;
            int src_x = (int)((px - x) * scale_x);
            if (src_x >= img_w)
                src_x = img_w - 1;
            int idx = row_offset + src_x * 3;

            int r = buffer[idx + 0];
            int g = buffer[idx + 1];
            int b = buffer[idx + 2];
            int color = (r << 16) | (g << 8) | b;
            double_buffer[py * screen_width + px] = color;
        }
    }
}

// 绘制文字（简单的文字显示，使用颜色块）
void draw_text_to_buffer(const char *text, int x, int y, int color)
{
    int arrow_x = x;
    int arrow_y = y + 20;
    int size = 20;

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size - i; j++)
        {
            int px = arrow_x + i;
            int py = arrow_y + j;
            if (px >= 0 && px < screen_width && py >= 0 && py < screen_height)
            {
                double_buffer[py * screen_width + px] = color;
            }
            px = arrow_x + i;
            py = arrow_y + size - 1 - j;
            if (px >= 0 && px < screen_width && py >= 0 && py < screen_height)
            {
                double_buffer[py * screen_width + px] = color;
            }
        }
    }
}

/* ============================================================ */
/* =============== YOLOv8 检测结果绘制接口 =============== */
/* 供 yolov8_detect.cpp 中 yolov8_draw_results() 调用（extern "C"）。
 * 注意：这两个函数操作的是屏幕/双缓冲坐标，调用方需保证传入的 x/y/w/h
 * 已经是“屏幕坐标系”下的值（在 vision_capture_thread_func 中已做好缩放转换）。
 */
/* ============================================================ */

/* ============================================================ */
/* =============== 简易 5x7 点阵字体（用于检测框类别文字标注） =============== */
/* 说明：工程未集成 freetype / 点阵字库，这里内置一套最小 5x7 像素字体，
 * 覆盖字母 A-Z（大小写共用同一套字形）、数字 0-9 和空格，
 * 足以显示 COCO 80 类标签（如 "person" "car" "cell phone" 等）。
 * 每个字符 7 行，每行低 5 位表示 5 列像素（bit4=最左列）。
 */
/* ============================================================ */
static const unsigned char font5x7_table[37][7] = {
    /* A */ {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11},
    /* B */ {0x1E, 0x11, 0x1E, 0x11, 0x11, 0x11, 0x1E},
    /* C */ {0x0F, 0x10, 0x10, 0x10, 0x10, 0x10, 0x0F},
    /* D */ {0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C},
    /* E */ {0x1F, 0x10, 0x1E, 0x10, 0x10, 0x10, 0x1F},
    /* F */ {0x1F, 0x10, 0x1E, 0x10, 0x10, 0x10, 0x10},
    /* G */ {0x0F, 0x10, 0x10, 0x17, 0x11, 0x11, 0x0F},
    /* H */ {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11},
    /* I */ {0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E},
    /* J */ {0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C},
    /* K */ {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11},
    /* L */ {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F},
    /* M */ {0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11},
    /* N */ {0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11},
    /* O */ {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E},
    /* P */ {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10},
    /* Q */ {0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D},
    /* R */ {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11},
    /* S */ {0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E},
    /* T */ {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04},
    /* U */ {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E},
    /* V */ {0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04},
    /* W */ {0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A},
    /* X */ {0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11},
    /* Y */ {0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04},
    /* Z */ {0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F},
    /* 0 */ {0x0E, 0x13, 0x15, 0x15, 0x19, 0x11, 0x0E},
    /* 1 */ {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E},
    /* 2 */ {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F},
    /* 3 */ {0x1E, 0x01, 0x01, 0x06, 0x01, 0x01, 0x1E},
    /* 4 */ {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02},
    /* 5 */ {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E},
    /* 6 */ {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E},
    /* 7 */ {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08},
    /* 8 */ {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E},
    /* 9 */ {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C},
    /* space / 不支持的字符 */ {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
};

// 将字符映射到 font5x7_table 的下标：A-Z/a-z -> 0~25，0-9 -> 26~35，其余(含空格) -> 36(空白)
static int font_char_index(char c)
{
    if (c >= 'A' && c <= 'Z')
        return c - 'A';
    if (c >= 'a' && c <= 'z')
        return c - 'a';
    if (c >= '0' && c <= '9')
        return 26 + (c - '0');
    return 36;
}

// 绘制单个字符：(x,y) 为字符左上角，scale 为像素放大倍数
static void draw_char_5x7(int x, int y, char c, int color, int scale)
{
    const unsigned char *glyph = font5x7_table[font_char_index(c)];
    for (int row = 0; row < 7; row++)
    {
        unsigned char bits = glyph[row];
        for (int col = 0; col < 5; col++)
        {
            if (!(bits & (1 << (4 - col))))
                continue;
            for (int sy = 0; sy < scale; sy++)
            {
                int py = y + row * scale + sy;
                if (py < 0 || py >= screen_height)
                    continue;
                for (int sx = 0; sx < scale; sx++)
                {
                    int px = x + col * scale + sx;
                    if (px < 0 || px >= screen_width)
                        continue;
                    double_buffer[py * screen_width + px] = color;
                }
            }
        }
    }
}

// 绘制字符串：(x,y) 为文字左上角
static void draw_string_5x7(int x, int y, const char *text, int color, int scale)
{
    int cx = x;
    int advance = 5 * scale + scale; // 字符宽度 + 1 像素间距(按 scale 放大)
    for (const char *p = text; *p != '\0'; p++)
    {
        draw_char_5x7(cx, y, *p, color, scale);
        cx += advance;
    }
}

void lcd_draw_rect(int x, int y, int w, int h, int color)
{
    for (int i = x; i < x + w; i++)
    {
        if (i < 0 || i >= screen_width)
            continue;
        for (int t = 0; t < 2; t++)
        {
            if (y + t >= 0 && y + t < screen_height)
                double_buffer[(y + t) * screen_width + i] = color;
            if (y + h - t >= 0 && y + h - t < screen_height)
                double_buffer[(y + h - t) * screen_width + i] = color;
        }
    }
    for (int j = y; j < y + h; j++)
    {
        if (j < 0 || j >= screen_height)
            continue;
        for (int t = 0; t < 2; t++)
        {
            if (x + t >= 0 && x + t < screen_width)
                double_buffer[j * screen_width + x + t] = color;
            if (x + w - t >= 0 && x + w - t < screen_width)
                double_buffer[j * screen_width + x + w - t] = color;
        }
    }
}

// 类别标签：使用内置 5x7 点阵字体在检测框上方绘制真实文字（如 "person"）。
// 标签背景填充为该类别对应的框颜色，文字颜色根据背景亮度自动选择黑/白以保证对比度。
void lcd_draw_text(int x, int y, const char *text, int color)
{
    if (text == NULL || text[0] == '\0')
        return;

    const int scale = 2; // 字体放大倍数，1个字符占 (5*scale + scale) x (7*scale) 像素
    int char_w = 5 * scale + scale;
    int text_len = (int)strlen(text);

    int label_w = text_len * char_w + 4;
    int label_h = 7 * scale + 4;

    int ly = y - label_h;
    if (ly < 0)
        ly = 0;

    // 标签背景：使用检测框颜色填充，便于和框对应起来
    for (int j = ly; j < ly + label_h; j++)
    {
        if (j < 0 || j >= screen_height)
            continue;
        for (int i = x; i < x + label_w; i++)
        {
            if (i < 0 || i >= screen_width)
                continue;
            double_buffer[j * screen_width + i] = color;
        }
    }

    // 根据背景颜色亮度自动选择文字颜色（黑/白），保证在各种框颜色下都清晰可读
    int r = (color >> 16) & 0xFF;
    int g = (color >> 8) & 0xFF;
    int b = color & 0xFF;
    int brightness = (r * 299 + g * 587 + b * 114) / 1000;
    int text_color = (brightness > 150) ? 0x000000 : 0xFFFFFF;

    draw_string_5x7(x + 2, ly + 2, text, text_color, scale);
}

/* ============================================================ */
/* =============== 缩略图绘制 =============== */
/* ============================================================ */

void draw_thumbnail_to_buffer(int index, int x, int y, int width, int height)
{
    if (index < 0 || index >= image_count)
        return;

    load_thumbnail(index);

    if (!thumbnail_cache[index].loaded)
    {
        for (int py = y; py < y + height; py++)
        {
            if (py < 0 || py >= screen_height)
                continue;
            for (int px = x; px < x + width; px++)
            {
                if (px < 0 || px >= screen_width)
                    continue;
                double_buffer[py * screen_width + px] = 0x444444;
            }
        }
        return;
    }

    ThumbnailCache *thumb = &thumbnail_cache[index];

    float scale_x = (float)width / thumb->width;
    float scale_y = (float)height / thumb->height;
    float scale = (scale_x < scale_y) ? scale_x : scale_y;

    int draw_w = (int)(thumb->width * scale);
    int draw_h = (int)(thumb->height * scale);
    int offset_x = x + (width - draw_w) / 2;
    int offset_y = y + (height - draw_h) / 2;

    float src_scale_x = (float)thumb->width / draw_w;
    float src_scale_y = (float)thumb->height / draw_h;

    for (int py = offset_y; py < offset_y + draw_h; py++)
    {
        if (py < 0 || py >= screen_height)
            continue;
        int src_y = (int)((py - offset_y) * src_scale_y);
        if (src_y >= thumb->height)
            src_y = thumb->height - 1;
        int row_offset = src_y * thumb->width * 3;

        for (int px = offset_x; px < offset_x + draw_w; px++)
        {
            if (px < 0 || px >= screen_width)
                continue;
            int src_x = (int)((px - offset_x) * src_scale_x);
            if (src_x >= thumb->width)
                src_x = thumb->width - 1;
            int idx = row_offset + src_x * 3;

            int r = thumb->buffer[idx + 0];
            int g = thumb->buffer[idx + 1];
            int b = thumb->buffer[idx + 2];
            int color = (r << 16) | (g << 8) | b;
            double_buffer[py * screen_width + px] = color;
        }
    }

    for (int i = 0; i < 2; i++)
    {
        for (int px = x; px < x + width; px++)
        {
            if (px < 0 || px >= screen_width)
                continue;
            if (y + i >= 0 && y + i < screen_height)
                double_buffer[(y + i) * screen_width + px] = 0x888888;
            if (y + height - 1 - i >= 0 && y + height - 1 - i < screen_height)
                double_buffer[(y + height - 1 - i) * screen_width + px] = 0x888888;
        }
        for (int py = y; py < y + height; py++)
        {
            if (py < 0 || py >= screen_height)
                continue;
            if (x + i >= 0 && x + i < screen_width)
                double_buffer[py * screen_width + (x + i)] = 0x888888;
            if (x + width - 1 - i >= 0 && x + width - 1 - i < screen_width)
                double_buffer[py * screen_width + (x + width - 1 - i)] = 0x888888;
        }
    }
}

/* ============================================================ */
/* =============== 页面绘制函数 =============== */
/* ============================================================ */

void draw_lockscreen()
{
    clear_double_buffer();

    show_image_to_buffer(IMAGE_LOCKSCREEN, draw_normal_to_buffer);

    int icon_width = 60;
    int icon_height = 60;
    int icon_x = (screen_width - icon_width) / 2;
    int icon_y = screen_height - 100;

    draw_icon_to_buffer(IMAGE_ICON_SWIPE_UP, icon_x, icon_y, icon_width, icon_height);

    flush_double_buffer();
    printf("=== 锁屏页面 ===\n");
    printf("从底部向上滑动解锁\n");
}

void draw_home()
{
    clear_double_buffer();

    // 如果还没有缓存背景图，加载并缓存
    if (!home_cached)
    {
        int w, h;
        char *buffer = NULL;
        if (load_image(IMAGE_BACKGROUND, &w, &h, &buffer))
        {
            home_background_buffer = buffer;
            home_bg_width = w;
            home_bg_height = h;
            home_cached = true;
        }
    }

    // 使用缓存的背景图绘制
    if (home_cached && home_background_buffer != NULL)
    {
        // 计算显示区域
        int disp_w, disp_h, off_x, off_y;
        calculate_display_area(home_bg_width, home_bg_height, &disp_w, &disp_h, &off_x, &off_y);
        draw_scaled_image_to_buffer_with_offset_limited(home_background_buffer, home_bg_width, home_bg_height,
                                                        disp_w, disp_h, off_x, off_y, 1.0f,
                                                        0, 0, screen_width, screen_height);
    }

    // 绘制图标（直接从缓存绘制，无需重新加载）
    draw_icon_to_buffer(IMAGE_ICON_HOME_CAMERA, ICON_HOME_CAMERA_X, ICON_HOME_CAMERA_Y,
                        ICON_HOME_CAMERA_W, ICON_HOME_CAMERA_H);
    draw_icon_to_buffer(IMAGE_ICON_HOME_ALBUM, ICON_HOME_ALBUM_X, ICON_HOME_ALBUM_Y,
                        ICON_HOME_ALBUM_W, ICON_HOME_ALBUM_H);
    draw_icon_to_buffer(IMAGE_ICON_HOME_VISION, ICON_HOME_VISION_X, ICON_HOME_VISION_Y,
                        ICON_HOME_VISION_W, ICON_HOME_VISION_H);

    flush_double_buffer();
    printf("=== 主页 ===\n");
}

/* ============================================================ */
/* =============== 摄像头实时预览 / 抓拍 / 录制 =============== */
/* ============================================================ */

static void ensure_album_dir(void)
{
    mkdir(album_dir, 0777);
}

// 在已加载列表中查找路径对应的索引，找不到返回 -1
static int find_image_index(const char *path)
{
    for (int i = 0; i < image_count; i++)
    {
        if (strcmp(image_paths[i], path) == 0)
            return i;
    }
    return -1;
}

// 重新扫描相册目录（抓拍/录制后调用，保证新文件出现在相册中）
static void refresh_album_list(void)
{
    free_thumbnail_cache();
    scan_directory(album_dir, ".jpg");
}

// 保存一帧 JPEG 数据到相册目录
static int save_jpeg_frame(unsigned char *data, unsigned long size, char *out_path, size_t out_path_len)
{
    ensure_album_dir();
    time_t now = time(NULL);
    snprintf(out_path, out_path_len, "%s/IMG_%ld.jpg", album_dir, (long)now);

    int fd = open(out_path, O_RDWR | O_CREAT | O_TRUNC, 0777);
    if (fd == -1)
    {
        perror("保存抓拍图片失败");
        return -1;
    }
    write(fd, data, size);
    close(fd);
    return 0;
}

// 绘制摄像头页面固定图标（预览帧之上叠加）
void draw_camera_icons_overlay(void)
{
    draw_icon_to_buffer(IMAGE_ICON_CLOSE, ICON_CAMERA_CLOSE_X, ICON_CAMERA_CLOSE_Y,
                        ICON_CAMERA_CLOSE_W, ICON_CAMERA_CLOSE_H);
    draw_icon_to_buffer(IMAGE_ICON_CAPTURE, ICON_CAMERA_CAPTURE_X, ICON_CAMERA_CAPTURE_Y,
                        ICON_CAMERA_CAPTURE_W, ICON_CAMERA_CAPTURE_H);
    draw_icon_to_buffer(IMAGE_ICON_RECORD, ICON_CAMERA_RECORD_X, ICON_CAMERA_RECORD_Y,
                        ICON_CAMERA_RECORD_W, ICON_CAMERA_RECORD_H);
    draw_icon_to_buffer(IMAGE_ICON_EXIT, ICON_EXIT_X, ICON_EXIT_Y,
                        ICON_EXIT_W, ICON_EXIT_H);

    // 录制中红点提示
    if (record_flag)
    {
        int rx = ICON_CAMERA_RECORD_X - 20;
        int ry = ICON_CAMERA_RECORD_Y - 20;
        for (int j = 0; j < 16; j++)
            for (int i = 0; i < 16; i++)
            {
                int px = rx + i, py = ry + j;
                if (px >= 0 && px < screen_width && py >= 0 && py < screen_height)
                    double_buffer[py * screen_width + px] = 0xFF0000;
            }
    }

    // 最近抓拍缩略图（点击可查看相册图片）
    if (has_last_capture)
    {
        int w, h;
        char *buf = NULL;
        if (load_image(last_capture_path, &w, &h, &buf))
        {
            float sx = (float)ICON_CAMERA_THUMB_W / w;
            float sy = (float)ICON_CAMERA_THUMB_H / h;
            float s = (sx < sy) ? sx : sy;
            int dw = (int)(w * s), dh = (int)(h * s);
            int ox = ICON_CAMERA_THUMB_X + (ICON_CAMERA_THUMB_W - dw) / 2;
            int oy = ICON_CAMERA_THUMB_Y + (ICON_CAMERA_THUMB_H - dh) / 2;
            draw_scaled_image_to_buffer_with_offset_limited(buf, w, h, dw, dh, ox, oy, 1.0f,
                                                            ICON_CAMERA_THUMB_X, ICON_CAMERA_THUMB_Y,
                                                            ICON_CAMERA_THUMB_W, ICON_CAMERA_THUMB_H);
            free(buf);
        }
        // 缩略图白色边框
        for (int i = 0; i < 2; i++)
        {
            for (int px = ICON_CAMERA_THUMB_X; px < ICON_CAMERA_THUMB_X + ICON_CAMERA_THUMB_W; px++)
            {
                int y1 = ICON_CAMERA_THUMB_Y + i, y2 = ICON_CAMERA_THUMB_Y + ICON_CAMERA_THUMB_H - 1 - i;
                if (px >= 0 && px < screen_width)
                {
                    if (y1 >= 0 && y1 < screen_height)
                        double_buffer[y1 * screen_width + px] = 0xFFFFFF;
                    if (y2 >= 0 && y2 < screen_height)
                        double_buffer[y2 * screen_width + px] = 0xFFFFFF;
                }
            }
            for (int py = ICON_CAMERA_THUMB_Y; py < ICON_CAMERA_THUMB_Y + ICON_CAMERA_THUMB_H; py++)
            {
                int x1 = ICON_CAMERA_THUMB_X + i, x2 = ICON_CAMERA_THUMB_X + ICON_CAMERA_THUMB_W - 1 - i;
                if (py >= 0 && py < screen_height)
                {
                    if (x1 >= 0 && x1 < screen_width)
                        double_buffer[py * screen_width + x1] = 0xFFFFFF;
                    if (x2 >= 0 && x2 < screen_width)
                        double_buffer[py * screen_width + x2] = 0xFFFFFF;
                }
            }
        }
    }
}

// 开始录制：以「4字节帧长 + JPEG数据」逐帧追加的 motion-jpeg 格式保存
void start_record(void)
{
    if (record_flag)
        return;
    ensure_album_dir();
    time_t now = time(NULL);
    snprintf(record_path, sizeof(record_path), "%s/VID_%ld.mjpeg", album_dir, (long)now);
    record_fp = fopen(record_path, "wb");
    if (record_fp == NULL)
    {
        perror("创建录像文件失败");
        return;
    }
    record_flag = 1;
    printf("开始录制: %s\n", record_path);
}

// 停止录制，并生成一张与视频同名的 jpg 缩略图，便于相册展示
void stop_record(void)
{
    if (!record_flag)
        return;
    record_flag = 0;
    if (record_fp != NULL)
    {
        fclose(record_fp);
        record_fp = NULL;
    }
    printf("停止录制: %s\n", record_path);

    if (has_last_capture && strlen(record_path) > 6)
    {
        char thumb_path[MAX_PATH_LEN];
        snprintf(thumb_path, sizeof(thumb_path), "%.*s.jpg",
                 (int)(strlen(record_path) - 6), record_path);
        FILE *src = fopen(last_capture_path, "rb");
        FILE *dst = fopen(thumb_path, "wb");
        if (src && dst)
        {
            char buf[4096];
            size_t n;
            while ((n = fread(buf, 1, sizeof(buf), src)) > 0)
                fwrite(buf, 1, n, dst);
        }
        if (src)
            fclose(src);
        if (dst)
            fclose(dst);
    }
}

// 摄像头实时采集线程：负责画面预览、抓拍保存、录制写帧
void *camera_capture_thread_func(void *arg)
{
    (void)arg;
    pthread_detach(pthread_self());

    camera_capture_open("/dev/video9");
    camera_capture_init(&camera_jpeg_buf);
    camera_capture_start_capturing();

    while (camera_flag == 1)
    {
        camera_capture_get_frame(&camera_jpeg_buf);
        // YUYV -> JPEG 编码
        yuyv2jpeg(camera_jpeg_buf.start, camera_jpeg_buf.length, 90);

        // 录制：将编码后的帧追加写入文件（帧长前缀，便于回放时分帧）
        if (record_flag && record_fp != NULL)
        {
            uint32_t frame_size = (uint32_t)camera_jpeg_buf.length;
            fwrite(&frame_size, sizeof(frame_size), 1, record_fp);
            fwrite(camera_jpeg_buf.start, 1, camera_jpeg_buf.length, record_fp);
        }

        // 抓拍：保存当前帧为 jpg 并加入相册
        if (snap_flag == 1)
        {
            char path[MAX_PATH_LEN];
            if (save_jpeg_frame(camera_jpeg_buf.start, camera_jpeg_buf.length, path, sizeof(path)) == 0)
            {
                strncpy(last_capture_path, path, MAX_PATH_LEN - 1);
                last_capture_path[MAX_PATH_LEN - 1] = '\0';
                has_last_capture = 1;
                printf("已抓拍并保存到相册: %s\n", path);
            }
            snap_flag = 0;
        }

        // 解码当前帧并显示到预览区域（左侧 0~800 像素区域）
        int w, h;
        char *rgb_buf = NULL;
        if (read_JPEG_mem(camera_jpeg_buf.start, camera_jpeg_buf.length, &w, &h, &rgb_buf))
        {
            pthread_mutex_lock(&camera_lock);
            if (current_page == PAGE_CAMERA)
            {
                clear_double_buffer();

                int dw, dh, ox, oy;
                float sx = 800.0f / w, sy = (float)screen_height / h;
                float s = (sx < sy) ? sx : sy;
                dw = (int)(w * s);
                dh = (int)(h * s);
                ox = (800 - dw) / 2;
                oy = (screen_height - dh) / 2;

                draw_scaled_image_to_buffer_with_offset_limited(rgb_buf, w, h, dw, dh, ox, oy, 1.0f,
                                                                0, 0, 800, screen_height);
                draw_camera_icons_overlay();
                flush_double_buffer();
            }
            pthread_mutex_unlock(&camera_lock);
            free(rgb_buf);
        }
    }

    camera_capture_stop_capturing();
    camera_capture_uninit(&camera_jpeg_buf);
    camera_capture_close();
    camera_thread_running = 0;
    pthread_exit(NULL);
}

// 启动摄像头：打开设备并开启预览线程，camera_flag 置 1
void start_camera(void)
{
    if (camera_thread_running)
        return;
    camera_flag = 1;
    camera_thread_running = 1;
    pthread_create(&camera_tid, NULL, camera_capture_thread_func, NULL);
}

// 停止摄像头：结束线程、关闭设备；若正在录制则一并停止
void stop_camera(void)
{
    if (record_flag)
    {
        stop_record();
    }
    if (camera_flag == 1)
    {
        camera_flag = 0;
        int wait_ms = 0;
        while (camera_thread_running && wait_ms < 1000)
        {
            usleep(10000);
            wait_ms += 10;
        }
    }
}

// 关闭摄像头后调用：仅将预览区域(左侧800px)绘制为黑屏，图标仍保留显示，
// 页面停留在 PAGE_CAMERA，不跳转回首页。
void draw_camera_closed_screen(void)
{
    pthread_mutex_lock(&camera_lock);
    clear_double_buffer();

    for (int py = 0; py < screen_height; py++)
    {
        for (int px = 0; px < 800; px++)
        {
            double_buffer[py * screen_width + px] = 0;
        }
    }

    draw_camera_icons_overlay();
    flush_double_buffer();
    pthread_mutex_unlock(&camera_lock);
}

void draw_camera()
{
    clear_double_buffer();

    for (int py = 0; py < screen_height; py++)
    {
        for (int px = 0; px < 800; px++)
        {
            double_buffer[py * screen_width + px] = 0;
        }
    }

    draw_camera_icons_overlay();
    flush_double_buffer();

    // 打开摄像头，camera_flag 置 1，预览线程会持续刷新画面
    start_camera();

    printf("=== 相机页面 ===\n");
}

/* ============================================================ */
/* =============== 视觉识别页面 (YOLOv8) =============== */
/* ============================================================ */

// 程序启动时调用一次，加载 YOLOv8 RKNN 模型与标签
void yolo_init(void)
{
    g_yolo_ctx = yolov8_init(YOLO_MODEL_PATH, YOLO_LABELS_PATH);
    if (g_yolo_ctx == NULL)
    {
        printf("YOLOv8 模型加载失败: %s，视觉识别功能不可用\n", YOLO_MODEL_PATH);
    }
    else
    {
        printf("YOLOv8 模型加载成功: %s\n", YOLO_MODEL_PATH);
    }
}

// 程序退出时调用一次，释放推理资源
void yolo_release(void)
{
    if (g_yolo_ctx != NULL)
    {
        yolov8_release(g_yolo_ctx);
        g_yolo_ctx = NULL;
    }
}

// 绘制视觉识别页面固定图标
void draw_vision_icons_overlay(void)
{
    draw_icon_to_buffer(IMAGE_ICON_CLOSE, ICON_VISION_CLOSE_X, ICON_VISION_CLOSE_Y,
                        ICON_VISION_CLOSE_W, ICON_VISION_CLOSE_H);

    if (g_yolo_ctx == NULL)
    {
        // 模型未加载时，用红色色块在右上角提示
        for (int j = 10; j < 30; j++)
            for (int i = 10; i < 200 && i < screen_width; i++)
                if (j < screen_height)
                    double_buffer[j * screen_width + i] = 0xFF0000;
    }
}

// 视觉识别采集线程：采集摄像头帧 -> 解码为RGB -> YOLOv8推理 -> 画检测框并显示
void *vision_capture_thread_func(void *arg)
{
    (void)arg;
    pthread_detach(pthread_self());

    camera_capture_open("/dev/video9");
    camera_capture_init(&vision_jpeg_buf);
    camera_capture_start_capturing();

    int frame_idx = 0;
    yolov8_result_list_t results;
    memset(&results, 0, sizeof(results));
    int have_results = 0;

    while (vision_flag == 1)
    {
        camera_capture_get_frame(&vision_jpeg_buf);
        yuyv2jpeg(vision_jpeg_buf.start, vision_jpeg_buf.length, 90);

        int w, h;
        char *rgb_buf = NULL;
        if (!read_JPEG_mem(vision_jpeg_buf.start, vision_jpeg_buf.length, &w, &h, &rgb_buf))
        {
            continue;
        }

        // 每隔 VISION_DETECT_INTERVAL 帧才跑一次推理，减轻 CPU/NPU 压力
        if (g_yolo_ctx != NULL && (frame_idx % VISION_DETECT_INTERVAL) == 0)
        {
            if (yolov8_detect_rgb(g_yolo_ctx, (unsigned char *)rgb_buf, w, h, &results) == 0)
            {
                have_results = 1;
                if (results.count > 0)
                {
                    printf("视觉识别: 检测到 %d 个目标\n", results.count);
                }
            }
        }
        frame_idx++;

        pthread_mutex_lock(&vision_lock);
        if (current_page == PAGE_VISION)
        {
            clear_double_buffer();

            int dw, dh, ox, oy;
            float sx = 800.0f / w, sy = (float)screen_height / h;
            float s = (sx < sy) ? sx : sy;
            dw = (int)(w * s);
            dh = (int)(h * s);
            ox = (800 - dw) / 2;
            oy = (screen_height - dh) / 2;

            draw_scaled_image_to_buffer_with_offset_limited(rgb_buf, w, h, dw, dh, ox, oy, 1.0f,
                                                            0, 0, 800, screen_height);

            // 把检测框从原始图像坐标系（w,h）映射到显示坐标系（dw,dh @ ox,oy）后绘制
            if (have_results)
            {
                for (int i = 0; i < results.count; i++)
                {
                    yolov8_detect_result_t *r = &results.results[i];
                    if (r->confidence < VISION_CONF_THRESHOLD)
                        continue;

                    int rx = ox + (int)(r->x * s);
                    int ry = oy + (int)(r->y * s);
                    int rw = (int)(r->width * s);
                    int rh = (int)(r->height * s);

                    int color = 0xFF0000;
                    if (r->class_id == 0)
                        color = 0x00FF00; // person - 绿色
                    else if (r->class_id == 2)
                        color = 0x0000FF; // car - 蓝色
                    else if (r->class_id == 5)
                        color = 0xFFFF00; // bus - 黄色
                    else if (r->class_id == 7)
                        color = 0xFF00FF; // truck - 紫色

                    lcd_draw_rect(rx, ry, rw, rh, color);
                    lcd_draw_text(rx, ry, r->class_name, color);

                    printf("  -> %s %.1f%% @ (%d,%d,%d,%d)\n",
                           r->class_name, r->confidence * 100, rx, ry, rw, rh);
                }
            }

            draw_vision_icons_overlay();
            flush_double_buffer();
        }
        pthread_mutex_unlock(&vision_lock);
        free(rgb_buf);
    }

    camera_capture_stop_capturing();
    camera_capture_uninit(&vision_jpeg_buf);
    camera_capture_close();
    vision_thread_running = 0;
    pthread_exit(NULL);
}

// 启动视觉识别：打开摄像头并开启检测线程
void start_vision(void)
{
    if (vision_thread_running)
        return;
    vision_flag = 1;
    vision_thread_running = 1;
    pthread_create(&vision_tid, NULL, vision_capture_thread_func, NULL);
}

// 停止视觉识别：结束线程、关闭设备
void stop_vision(void)
{
    if (vision_flag == 1)
    {
        vision_flag = 0;
        int wait_ms = 0;
        while (vision_thread_running && wait_ms < 1000)
        {
            usleep(10000);
            wait_ms += 10;
        }
    }
}

void draw_vision(void)
{
    clear_double_buffer();

    for (int py = 0; py < screen_height; py++)
    {
        for (int px = 0; px < 800; px++)
        {
            double_buffer[py * screen_width + px] = 0;
        }
    }

    draw_vision_icons_overlay();
    flush_double_buffer();

    // 打开摄像头并开始持续检测，预览线程会持续刷新画面
    start_vision();

    printf("=== 视觉识别页面 ===\n");
    if (g_yolo_ctx == NULL)
    {
        printf("警告: YOLOv8 模型未成功加载，当前仅显示画面，无法绘制检测框\n");
    }
}

void draw_album()
{
    clear_double_buffer();

    for (int py = 0; py < screen_height; py++)
    {
        for (int px = 0; px < screen_width; px++)
        {
            double_buffer[py * screen_width + px] = 0x1a1a1a;
        }
    }

    clamp_album_scroll();

    int total_images = image_count;
    if (total_images > 0)
    {
        int total_rows = (total_images + THUMB_COLS - 1) / THUMB_COLS;
        for (int i = 0; i < total_rows; i++)
        {
            int y = THUMB_TOP + i * (THUMB_HEIGHT + THUMB_GAP) - album_scroll_offset;

            // 整行都在可视区域之外，跳过
            if (y + THUMB_HEIGHT <= ALBUM_VIEW_TOP || y >= ALBUM_VIEW_BOTTOM)
                continue;

            for (int j = 0; j < THUMB_COLS && i * THUMB_COLS + j < total_images; j++)
            {
                int idx = i * THUMB_COLS + j;
                int x = THUMB_LEFT + j * (THUMB_WIDTH + THUMB_GAP);
                if (x + THUMB_WIDTH > 1000)
                    break;
                draw_thumbnail_to_buffer_clipped(idx, x, y, THUMB_WIDTH, THUMB_HEIGHT,
                                                  ALBUM_VIEW_TOP, ALBUM_VIEW_BOTTOM);
            }
        }

        // 内容超出可视区域时，绘制右侧滚动条
        int content_height, max_scroll;
        get_album_scroll_bounds(&content_height, &max_scroll);
        if (max_scroll > 0)
        {
            int viewport_h = ALBUM_VIEW_BOTTOM - ALBUM_VIEW_TOP;

            // 滚动条轨道
            for (int py = ALBUM_VIEW_TOP; py < ALBUM_VIEW_BOTTOM; py++)
            {
                for (int px = ALBUM_SCROLLBAR_X; px < ALBUM_SCROLLBAR_X + ALBUM_SCROLLBAR_WIDTH; px++)
                {
                    if (px >= 0 && px < screen_width && py >= 0 && py < screen_height)
                        double_buffer[py * screen_width + px] = 0x333333;
                }
            }

            // 滚动条滑块
            int thumb_h = viewport_h * viewport_h / content_height;
            if (thumb_h < 20)
                thumb_h = 20;
            if (thumb_h > viewport_h)
                thumb_h = viewport_h;
            int thumb_y = ALBUM_VIEW_TOP +
                          (int)((long)album_scroll_offset * (viewport_h - thumb_h) / max_scroll);

            for (int py = thumb_y; py < thumb_y + thumb_h; py++)
            {
                for (int px = ALBUM_SCROLLBAR_X; px < ALBUM_SCROLLBAR_X + ALBUM_SCROLLBAR_WIDTH; px++)
                {
                    if (px >= 0 && px < screen_width && py >= 0 && py < screen_height)
                        double_buffer[py * screen_width + px] = 0xAAAAAA;
                }
            }
        }
    }
    else
    {
        printf("相册中没有图片!\n");
    }

    draw_icon_to_buffer(IMAGE_ICON_EXIT, ICON_ALBUM_EXIT_X, ICON_ALBUM_EXIT_Y,
                        ICON_ALBUM_EXIT_W, ICON_ALBUM_EXIT_H);

    char info_text[64];
    snprintf(info_text, sizeof(info_text), "图片: %d", total_images);
    printf("%s\n", info_text);

    flush_double_buffer();
    printf("=== 相册页面(缩略图，可上下滑动) ===\n");
}

void draw_icons_overlay(void);

// 使用当前选中的特效显示图片
void draw_view_with_effect(bool fast_switch)
{
    clear_double_buffer();

    // 绘制黑色背景（全屏）
    for (int py = 0; py < screen_height; py++)
    {
        for (int px = 0; px < screen_width; px++)
        {
            double_buffer[py * screen_width + px] = 0;
        }
    }

    if (image_count > 0 && current_image_index < image_count)
    {
        const char *current_image = image_paths[current_image_index];

        if (lcd_load_image(current_image) == 0)
        {
            draw_callback_t effect_func = effect_functions[current_effect_index % EFFECT_COUNT];

            if (fast_switch || current_effect_index == 0)
            {
                // 快速切换：直接显示，不播放特效动画
                calculate_display_area(current_width, current_height, &g_display_w, &g_display_h, &g_display_x, &g_display_y);
                clear_double_buffer();
                effect_func(current_buffer, current_width, current_height, g_display_w, g_display_h, g_display_x, g_display_y);
            }
            else
            {
                // 正常模式：播放特效动画
                draw_current_image_to_buffer(effect_func);
            }
        }
    }
    else
    {
        printf("没有图片可显示!\n");
    }

    // 绘制所有图标（在图片上层）- 始终显示
    draw_icons_overlay();

    flush_double_buffer();
    printf("=== 全屏查看模式 ===\n");
    printf("图片 %d/%d, 特效: %s, 缩放: %.2fx\n", current_image_index + 1, image_count,
           effect_names[current_effect_index % EFFECT_COUNT], current_scale);
}

// 带垂直裁剪范围的缩略图绘制（用于可滑动的相册区域，超出 clip_top/clip_bottom 的部分不绘制）
void draw_thumbnail_to_buffer_clipped(int index, int x, int y, int width, int height,
                                       int clip_top, int clip_bottom)
{
    if (index < 0 || index >= image_count)
        return;
    if (y + height <= clip_top || y >= clip_bottom)
        return; // 完全在可视区域之外
 
    load_thumbnail(index);
 
    if (!thumbnail_cache[index].loaded)
    {
        for (int py = y; py < y + height; py++)
        {
            if (py < 0 || py >= screen_height || py < clip_top || py >= clip_bottom)
                continue;
            for (int px = x; px < x + width; px++)
            {
                if (px < 0 || px >= screen_width)
                    continue;
                double_buffer[py * screen_width + px] = 0x444444;
            }
        }
        return;
    }
 
    ThumbnailCache *thumb = &thumbnail_cache[index];
 
    float scale_x = (float)width / thumb->width;
    float scale_y = (float)height / thumb->height;
    float scale = (scale_x < scale_y) ? scale_x : scale_y;
 
    int draw_w = (int)(thumb->width * scale);
    int draw_h = (int)(thumb->height * scale);
    int offset_x = x + (width - draw_w) / 2;
    int offset_y = y + (height - draw_h) / 2;
 
    float src_scale_x = (float)thumb->width / draw_w;
    float src_scale_y = (float)thumb->height / draw_h;
 
    for (int py = offset_y; py < offset_y + draw_h; py++)
    {
        if (py < 0 || py >= screen_height || py < clip_top || py >= clip_bottom)
            continue;
        int src_y = (int)((py - offset_y) * src_scale_y);
        if (src_y >= thumb->height)
            src_y = thumb->height - 1;
        int row_offset = src_y * thumb->width * 3;
 
        for (int px = offset_x; px < offset_x + draw_w; px++)
        {
            if (px < 0 || px >= screen_width)
                continue;
            int src_x = (int)((px - offset_x) * src_scale_x);
            if (src_x >= thumb->width)
                src_x = thumb->width - 1;
            int idx = row_offset + src_x * 3;
 
            int r = thumb->buffer[idx + 0];
            int g = thumb->buffer[idx + 1];
            int b = thumb->buffer[idx + 2];
            int color = (r << 16) | (g << 8) | b;
            double_buffer[py * screen_width + px] = color;
        }
    }
 
    for (int i = 0; i < 2; i++)
    {
        for (int px = x; px < x + width; px++)
        {
            if (px < 0 || px >= screen_width)
                continue;
            int yy1 = y + i, yy2 = y + height - 1 - i;
            if (yy1 >= 0 && yy1 < screen_height && yy1 >= clip_top && yy1 < clip_bottom)
                double_buffer[yy1 * screen_width + px] = 0x888888;
            if (yy2 >= 0 && yy2 < screen_height && yy2 >= clip_top && yy2 < clip_bottom)
                double_buffer[yy2 * screen_width + px] = 0x888888;
        }
        for (int py = y; py < y + height; py++)
        {
            if (py < 0 || py >= screen_height || py < clip_top || py >= clip_bottom)
                continue;
            if (x + i >= 0 && x + i < screen_width)
                double_buffer[py * screen_width + (x + i)] = 0x888888;
            if (x + width - 1 - i >= 0 && x + width - 1 - i < screen_width)
                double_buffer[py * screen_width + (x + width - 1 - i)] = 0x888888;
        }
    }
}

// 计算相册总行数/内容高度/最大可滚动距离
void get_album_scroll_bounds(int *content_height, int *max_scroll)
{
    int total_rows = (image_count + THUMB_COLS - 1) / THUMB_COLS;
    if (total_rows < 0)
        total_rows = 0;
    int ch = total_rows > 0 ? total_rows * (THUMB_HEIGHT + THUMB_GAP) - THUMB_GAP : 0;
    int viewport_h = ALBUM_VIEW_BOTTOM - ALBUM_VIEW_TOP;
    int ms = ch - viewport_h;
    if (ms < 0)
        ms = 0;
    if (content_height) *content_height = ch;
    if (max_scroll) *max_scroll = ms;
}
 
// 将滚动偏移限制在合法范围内
void clamp_album_scroll(void)
{
    int content_height, max_scroll;
    get_album_scroll_bounds(&content_height, &max_scroll);
    if (album_scroll_offset < 0)
        album_scroll_offset = 0;
    if (album_scroll_offset > max_scroll)
        album_scroll_offset = max_scroll;
}

// 提取图标绘制为独立函数，便于复用
void draw_icons_overlay()
{
    draw_icon_to_buffer(IMAGE_ICON_PREV, ICON_VIEW_LEFT_X, ICON_VIEW_LEFT_Y,
                        ICON_VIEW_LEFT_W, ICON_VIEW_LEFT_H);
    draw_icon_to_buffer(IMAGE_ICON_NEXT, ICON_VIEW_RIGHT_X, ICON_VIEW_RIGHT_Y,
                        ICON_VIEW_RIGHT_W, ICON_VIEW_RIGHT_H);
    draw_icon_to_buffer(IMAGE_ICON_EXIT, ICON_VIEW_EXIT_X, ICON_VIEW_EXIT_Y,
                        ICON_VIEW_EXIT_W, ICON_VIEW_EXIT_H);
    draw_icon_to_buffer(IMAGE_ICON_EFFECT, ICON_VIEW_EFFECT_X, ICON_VIEW_EFFECT_Y,
                        ICON_VIEW_EFFECT_W, ICON_VIEW_EFFECT_H);
    draw_icon_to_buffer(IMAGE_ICON_ZOOM_IN, ICON_VIEW_ZOOM_IN_X, ICON_VIEW_ZOOM_IN_Y,
                        ICON_VIEW_ZOOM_IN_W, ICON_VIEW_ZOOM_IN_H);
    draw_icon_to_buffer(IMAGE_ICON_ZOOM_OUT, ICON_VIEW_ZOOM_OUT_X, ICON_VIEW_ZOOM_OUT_Y,
                        ICON_VIEW_ZOOM_OUT_W, ICON_VIEW_ZOOM_OUT_H);
}

// 为了兼容性，保留原函数名
void draw_view()
{
    draw_view_with_effect(true);
}

/* ============================================================ */
/* =============== 触摸事件处理 =============== */
/* ============================================================ */

int get_thumbnail_index(int x, int y)
{
    if (y < ALBUM_VIEW_TOP || y >= ALBUM_VIEW_BOTTOM)
        return -1;

    int total_rows = (image_count + THUMB_COLS - 1) / THUMB_COLS;
    for (int i = 0; i < total_rows; i++)
    {
        int ty = THUMB_TOP + i * (THUMB_HEIGHT + THUMB_GAP) - album_scroll_offset;

        if (ty + THUMB_HEIGHT <= ALBUM_VIEW_TOP || ty >= ALBUM_VIEW_BOTTOM)
            continue;

        for (int j = 0; j < THUMB_COLS; j++)
        {
            int idx = i * THUMB_COLS + j;
            if (idx >= image_count)
                return -1;

            int tx = THUMB_LEFT + j * (THUMB_WIDTH + THUMB_GAP);

            if (is_in_region(x, y, tx, ty, THUMB_WIDTH, THUMB_HEIGHT))
            {
                return idx;
            }
        }
    }
    return -1;
}

// 处理缩放
void handle_zoom(float scale_factor)
{
    current_scale *= scale_factor;
    if (current_scale < min_scale)
        current_scale = min_scale;
    if (current_scale > max_scale)
        current_scale = max_scale;

    if (current_page == PAGE_VIEW && current_buffer != NULL)
    {
        clear_double_buffer();

        for (int py = 0; py < screen_height; py++)
        {
            for (int px = 0; px < screen_width; px++)
            {
                double_buffer[py * screen_width + px] = 0;
            }
        }

        draw_callback_t effect_func = effect_functions[current_effect_index % EFFECT_COUNT];

        calculate_display_area(current_width, current_height, &g_display_w, &g_display_h, &g_display_x, &g_display_y);

        clear_double_buffer();
        effect_func(current_buffer, current_width, current_height, g_display_w, g_display_h, g_display_x, g_display_y);

        draw_icons_overlay();

        flush_double_buffer();
        printf("缩放: %.2fx\n", current_scale);
    }
}

// 处理两指手势
void handle_pinch_gesture(TouchPoint p1, TouchPoint p2, float last_distance)
{
    float current_distance = get_two_finger_distance(p1, p2);
    if (last_distance > 0)
    {
        float ratio = current_distance / last_distance;
        if (ratio > 1.05f || ratio < 0.95f)
        {
            handle_zoom(ratio);
        }
    }
}

void handle_touch_event(TouchPoint point)
{
    if (!point.valid)
        return;

    int x = point.x;
    int y = point.y;

    printf("触摸: X=%d, Y=%d\n", x, y);

    switch (current_page)
    {
    case PAGE_LOCKSCREEN:
        if (y < 100)
        {
            printf("解锁中...\n");
            current_page = PAGE_HOME;
            draw_home();
        }
        break;

    case PAGE_HOME:
        if (is_in_region(x, y, ICON_HOME_CAMERA_X, ICON_HOME_CAMERA_Y,
                         ICON_HOME_CAMERA_W, ICON_HOME_CAMERA_H))
        {
            printf("进入相机\n");
            current_page = PAGE_CAMERA;
            draw_camera();
        }
        else if (is_in_region(x, y, ICON_HOME_ALBUM_X, ICON_HOME_ALBUM_Y,
                              ICON_HOME_ALBUM_W, ICON_HOME_ALBUM_H))
        {
            printf("进入相册\n");
            current_page = PAGE_ALBUM;
            load_all_thumbnails();
            draw_album();
        }
        else if (is_in_region(x, y, ICON_HOME_VISION_X, ICON_HOME_VISION_Y,
                              ICON_HOME_VISION_W, ICON_HOME_VISION_H))
        {
            printf("进入视觉识别\n");
            current_page = PAGE_VISION;
            draw_vision();
        }
        break;

    case PAGE_CAMERA:
        if (is_in_region(x, y, ICON_CAMERA_CLOSE_X, ICON_CAMERA_CLOSE_Y,
                         ICON_CAMERA_CLOSE_W, ICON_CAMERA_CLOSE_H))
        {
            // 判断当前摄像头状态
            if (camera_flag == 0) 
            {
                // 如果当前是关闭状态，再次点击则“重新开启”
                printf("重新开启摄像头预览\n");
                start_camera();
                // 注意：这里可能需要调用你原本负责刷新正常预览画面的函数，例如 draw_camera_preview();
            }
            else 
            {
                // 如果当前是开启状态，点击则“关闭(黑屏)”
                printf("关闭摄像头(黑屏)\n");
                stop_camera();
                draw_camera_closed_screen();
            }
        }
        else if (is_in_region(x, y, ICON_CAMERA_CAPTURE_X, ICON_CAMERA_CAPTURE_Y,
                              ICON_CAMERA_CAPTURE_W, ICON_CAMERA_CAPTURE_H))
        {
            if (camera_flag == 0)
            {
                // 之前已黑屏关闭，先重新开启预览再拍照
                printf("摄像头已关闭，重新开启预览\n");
                start_camera();
            }
            printf("拍照\n");
            snap_flag = 1;
        }
        else if (is_in_region(x, y, ICON_CAMERA_RECORD_X, ICON_CAMERA_RECORD_Y,
                              ICON_CAMERA_RECORD_W, ICON_CAMERA_RECORD_H))
        {
            if (camera_flag == 0)
            {
                printf("摄像头已关闭，重新开启预览\n");
                start_camera();
            }
            if (record_flag)
            {
                printf("停止录像\n");
                stop_record();
            }
            else
            {
                printf("开始录像\n");
                start_record();
            }
        }
        else if (has_last_capture && is_in_region(x, y, ICON_CAMERA_THUMB_X, ICON_CAMERA_THUMB_Y,
                                                  ICON_CAMERA_THUMB_W, ICON_CAMERA_THUMB_H))
        {
            printf("查看最近抓拍\n");
            stop_camera();
            refresh_album_list();
            load_all_thumbnails();
            int idx = find_image_index(last_capture_path);
            current_image_index = (idx >= 0) ? idx : 0;
            current_scale = 1.0f;
            current_page = PAGE_VIEW;
            draw_view_with_effect(true);
        }
        else if (is_in_region(x, y, ICON_EXIT_X, ICON_EXIT_Y,
                              ICON_EXIT_W, ICON_EXIT_H))
        {
            printf("退出相机\n");
            stop_camera();
            current_page = PAGE_HOME;
            draw_home();
        }
        break;

    case PAGE_ALBUM:
    {
        int swipe_delta = point.start_y - point.y; // 正值表示手指向上滑动（查看下方内容）

        if (abs(swipe_delta) > 15)
        {
            // 识别为滑动手势：滚动缩略图列表，不触发点击
            album_scroll_offset += swipe_delta;
            clamp_album_scroll();
            printf("相册滑动: offset=%d\n", album_scroll_offset);
            draw_album();
            break;
        }

        int thumb_idx = get_thumbnail_index(x, y);
        if (thumb_idx >= 0 && thumb_idx < image_count)
        {
            current_image_index = thumb_idx;
            current_scale = 1.0f;
            printf("选择图片: %d\n", current_image_index);
            current_page = PAGE_VIEW;
            draw_view_with_effect(true);
            break;
        }
    }

        if (is_in_region(x, y, ICON_ALBUM_EXIT_X, ICON_ALBUM_EXIT_Y,
                         ICON_ALBUM_EXIT_W, ICON_ALBUM_EXIT_H))
        {
            printf("退出相册\n");
            free_thumbnail_cache();
            current_page = PAGE_HOME;
            draw_home();
        }
        break;


    case PAGE_VIEW:
        if (is_in_region(x, y, ICON_VIEW_ZOOM_IN_X, ICON_VIEW_ZOOM_IN_Y,
                         ICON_VIEW_ZOOM_IN_W, ICON_VIEW_ZOOM_IN_H))
        {
            printf("放大图片\n");
            handle_zoom(1.2f);
        }
        else if (is_in_region(x, y, ICON_VIEW_ZOOM_OUT_X, ICON_VIEW_ZOOM_OUT_Y,
                              ICON_VIEW_ZOOM_OUT_W, ICON_VIEW_ZOOM_OUT_H))
        {
            printf("缩小图片\n");
            handle_zoom(0.8f);
        }
        else if (is_in_region(x, y, ICON_VIEW_EXIT_X, ICON_VIEW_EXIT_Y,
                              ICON_VIEW_EXIT_W, ICON_VIEW_EXIT_H))
        {
            printf("返回相册缩略图\n");
            current_page = PAGE_ALBUM;
            load_all_thumbnails();
            draw_album();
        }
        else if (is_in_region(x, y, ICON_VIEW_EFFECT_X, ICON_VIEW_EFFECT_Y,
                              ICON_VIEW_EFFECT_W, ICON_VIEW_EFFECT_H))
        {
            current_effect_index = (current_effect_index + 1) % EFFECT_COUNT;
            printf("切换特效: %s\n", effect_names[current_effect_index]);
            draw_view_with_effect(false);
        }
        else if (is_in_region(x, y, ICON_VIEW_LEFT_X, ICON_VIEW_LEFT_Y,
                              ICON_VIEW_LEFT_W, ICON_VIEW_LEFT_H))
        {
            current_image_index = (current_image_index - 1 + image_count) % image_count;
            current_scale = 1.0f;
            current_effect_index = random() % EFFECT_COUNT;
            printf("上一张: %d, 随机特效: %s\n", current_image_index, effect_names[current_effect_index]);
            draw_view_with_effect(false);
        }
        else if (is_in_region(x, y, ICON_VIEW_RIGHT_X, ICON_VIEW_RIGHT_Y,
                              ICON_VIEW_RIGHT_W, ICON_VIEW_RIGHT_H))
        {
            current_image_index = (current_image_index + 1) % image_count;
            current_scale = 1.0f;
            current_effect_index = random() % EFFECT_COUNT;
            printf("下一张: %d, 随机特效: %s\n", current_image_index, effect_names[current_effect_index]);
            draw_view_with_effect(false);
        }
        break;

    case PAGE_VISION:
        if (is_in_region(x, y, ICON_VISION_CLOSE_X, ICON_VISION_CLOSE_Y,
                         ICON_VISION_CLOSE_W, ICON_VISION_CLOSE_H))
        {
            printf("关闭视觉识别\n");
            stop_vision();
            current_page = PAGE_HOME;
            draw_home();
        }
        break;

    default:
        break;
    }
}

/* ============================================================ */
/* =============== 主函数 =============== */
/* ============================================================ */

int main(int argc, char **argv)
{
    if (dev_init() != 0)
    {
        printf("LCD 初始化失败!\n");
        return -1;
    }

    // 预加载常用图标到缓存
    get_cached_icon(IMAGE_ICON_PREV, NULL, NULL);
    get_cached_icon(IMAGE_ICON_NEXT, NULL, NULL);
    get_cached_icon(IMAGE_ICON_EXIT, NULL, NULL);
    get_cached_icon(IMAGE_ICON_EFFECT, NULL, NULL);
    get_cached_icon(IMAGE_ICON_ZOOM_IN, NULL, NULL);
    get_cached_icon(IMAGE_ICON_ZOOM_OUT, NULL, NULL);
    get_cached_icon(IMAGE_ICON_SWIPE_UP, NULL, NULL);
    get_cached_icon(IMAGE_ICON_HOME_CAMERA, NULL, NULL);
    get_cached_icon(IMAGE_ICON_HOME_ALBUM, NULL, NULL);
    get_cached_icon(IMAGE_ICON_HOME_VISION, NULL, NULL);

    srandom(time(NULL));

    // 加载 YOLOv8 模型（一次性初始化，供"视觉识别"页面复用）
    yolo_init();

    char scan_dir[MAX_PATH_LEN] = "./photos";
    char file_type[MAX_PATH_LEN] = ".jpg";

    if (argc > 1)
    {
        strncpy(scan_dir, argv[1], MAX_PATH_LEN - 1);
    }
    if (argc > 2)
    {
        strncpy(file_type, argv[2], MAX_PATH_LEN - 1);
        if (file_type[0] != '.')
        {
            char temp[MAX_PATH_LEN];
            strcpy(temp, file_type);
            sprintf(file_type, ".%s", temp);
        }
    }

    int found = scan_directory(scan_dir, file_type);
    strncpy(album_dir, scan_dir, MAX_PATH_LEN - 1);
    album_dir[MAX_PATH_LEN - 1] = '\0';
    if (found > 0)
    {
        print_image_list();
    }
    else
    {
        printf("在目录 %s 中未找到 %s 文件\n", scan_dir, file_type);
        printf("继续运行(无图片)...\n");
    }

    draw_lockscreen();

    printf("\n=== 系统启动 ===\n");
    printf("按 Ctrl+C 退出\n");
    printf("从底部向上滑动解锁\n\n");

    while (1)
    {
        TouchPoint point = get_touch_data(ts_fd);
        if (point.valid)
        {
            handle_touch_event(point);
        }
        usleep(10000);
    }

    release_current_image();
    dev_uninit();
    return 0;
}