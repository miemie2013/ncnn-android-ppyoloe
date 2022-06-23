// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "ppyoloe.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"




struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}


static void generate_ppyoloe_proposals(const ncnn::Mat& cls_score, const ncnn::Mat& reg_dist, float scale_x, float scale_y, float prob_threshold, std::vector<Object>& objects)
{
    // python中cls_score的形状是[N, A, 80], ncnn中C=1, H=A=预测框数, W=80
    // python中reg_dist 的形状是[N, A,  4], ncnn中C=1, H=A=预测框数, W= 4
    int C = cls_score.c;
    int H = cls_score.h;
    int W = cls_score.w;
//    printf("C=%d\n", C);
//    printf("H=%d\n", H);
//    printf("W=%d\n", W);
    int num_grid = H;
    int num_class = W;

    // 最大感受野输出的特征图一行（一列）的格子数stride32_grid。设为G，则
    // G*G + (2*G)*(2*G) + (4*G)*(4*G) = 21*G^2 = W
    // 所以G = sqrt(W/21)
    int stride32_grid = sqrt(num_grid / 21);
    int stride16_grid = stride32_grid * 2;
    int stride8_grid = stride32_grid * 4;

    // 因为二者的C都只等于1，所以取第0个
    const float* cls_score_ptr = cls_score.channel(0);
    const float* reg_dist_ptr = reg_dist.channel(0);

    // stride==32的格子结束的位置
    int stride32_end = stride32_grid * stride32_grid;
    // stride==16的格子结束的位置
    int stride16_end = stride32_grid * stride32_grid * 5;
    for (int anchor_idx = 0; anchor_idx < num_grid; anchor_idx++)
    {
        float stride = 32.0f;
        int row_i = 0;
        int col_i = 0;
        if (anchor_idx < stride32_end) {
            stride = 32.0f;
            row_i = anchor_idx / stride32_grid;
            col_i = anchor_idx % stride32_grid;
        }else if (anchor_idx < stride16_end) {
            stride = 16.0f;
            row_i = (anchor_idx - stride32_end) / stride16_grid;
            col_i = (anchor_idx - stride32_end) % stride16_grid;
        }else {  // stride == 8
            stride = 8.0f;
            row_i = (anchor_idx - stride16_end) / stride8_grid;
            col_i = (anchor_idx - stride16_end) % stride8_grid;
        }
        float x_center = 0.5f + (float)col_i;
        float y_center = 0.5f + (float)row_i;
        float x0 = x_center - reg_dist_ptr[0];
        float y0 = y_center - reg_dist_ptr[1];
        float x1 = x_center + reg_dist_ptr[2];
        float y1 = y_center + reg_dist_ptr[3];
        x0 = x0 * stride / scale_x;
        y0 = y0 * stride / scale_y;
        x1 = x1 * stride / scale_x;
        y1 = y1 * stride / scale_y;
        float h = y1 - y0;
        float w = x1 - x0;

        for (int class_idx = 0; class_idx < num_class; class_idx++)
        {
            float box_prob = cls_score_ptr[class_idx];
            if (box_prob > prob_threshold)
            {
                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = class_idx;
                obj.prob = box_prob;

                objects.push_back(obj);
            }

        } // class loop
        cls_score_ptr += cls_score.w;
        reg_dist_ptr += reg_dist.w;
    }
}


PPYOLOE::PPYOLOE()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int PPYOLOE::load(const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    model.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    model.opt = ncnn::Option();

#if NCNN_VULKAN
    model.opt.use_vulkan_compute = use_gpu;
#endif
    model.opt.num_threads = ncnn::get_big_cpu_count();
    model.opt.blob_allocator = &blob_pool_allocator;
    model.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);

    model.load_param(parampath);
    model.load_model(modelpath);

    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

int PPYOLOE::load(AAssetManager* mgr, const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    model.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    model.opt = ncnn::Option();
#if NCNN_VULKAN
    model.opt.use_vulkan_compute = use_gpu;
#endif
    model.opt.num_threads = ncnn::get_big_cpu_count();
    model.opt.blob_allocator = &blob_pool_allocator;
    model.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);

    model.load_param(mgr, parampath);
    model.load_model(mgr, modelpath);


    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}


int PPYOLOE::detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold, float nms_threshold)
{

    int img_w = rgb.cols;
    int img_h = rgb.rows;
    float scale_x = (float)target_size / img_w;
    float scale_y = (float)target_size / img_h;

    // get ncnn::Mat with RGB format like PPYOLOE do.
    ncnn::Mat in_rgb = ncnn::Mat::from_pixels(rgb.data, ncnn::Mat::PIXEL_RGB, img_w, img_h);
    ncnn::Mat in_resize;
    // Interp image with cv2.INTER_CUBIC like PPYOLOE do.
    ncnn::resize_bicubic(in_rgb, in_resize, target_size, target_size);

    // Normalize image with the same mean and std like PPYOLOE do.
    in_resize.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = model.create_extractor();

    ex.input("images", in_resize);

    std::vector<Object> proposals;

    {
        ncnn::Mat cls_score;  // python中的形状是[N, A, 80], ncnn中C=1, H=A=预测框数, W=80
        ncnn::Mat reg_dist;   // python中的形状是[N, A,  4], ncnn中C=1, H=A=预测框数, W= 4
        ex.extract("cls_score", cls_score);
        ex.extract("reg_dist", reg_dist);
        generate_ppyoloe_proposals(cls_score, reg_dist, scale_x, scale_y, prob_threshold, proposals);
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        //
        float x0 = (objects[i].rect.x);
        float y0 = (objects[i].rect.y);
        float x1 = (objects[i].rect.x + objects[i].rect.width);
        float y1 = (objects[i].rect.y + objects[i].rect.height);

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    return 0;
}

int PPYOLOE::draw(cv::Mat& rgb, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
    };
    static const unsigned char colors[19][3] = {
        { 54,  67, 244},
        { 99,  30, 233},
        {176,  39, 156},
        {183,  58, 103},
        {181,  81,  63},
        {243, 150,  33},
        {244, 169,   3},
        {212, 188,   0},
        {136, 150,   0},
        { 80, 175,  76},
        { 74, 195, 139},
        { 57, 220, 205},
        { 59, 235, 255},
        {  7, 193, 255},
        {  0, 152, 255},
        { 34,  87, 255},
        { 72,  85, 121},
        {158, 158, 158},
        {139, 125,  96}
    };

    int color_index = 0;

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        const unsigned char* color = colors[color_index % 19];
        color_index++;

        cv::Scalar cc(color[0], color[1], color[2]);

        cv::rectangle(rgb,obj.rect, cc, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cc, -1);

        cv::Scalar textcc = (color[0] + color[1] + color[2] >= 381) ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);

        cv::putText(rgb, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, textcc, 1);

    }
    
    
    return 0;
}
