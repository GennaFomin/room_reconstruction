#include "opencv2/opencv.hpp"

// blurs rect of image with kernel of sha[e size]
void blur_part(cv::Mat& image, cv::Rect roi, int shape = 100) {
    cv::Mat kernel = cv::Mat::ones(shape, shape, CV_32F) / float(shape * shape);
    cv::Mat swap;
    image(roi).copyTo(swap);
    cv::filter2D(swap, swap, -1, kernel);
    swap.copyTo(image(roi));
}

// Pads image with blurred replicated border
cv::Mat pad_image(const cv::Mat& image, int left, int shape = 100) {
    int h = image.size[0];
    int w = image.size[1];
    int padding = std::max(0, ((w + left) / 2 - h) / 2);
    cv::Mat prom;
    cv::copyMakeBorder(image, prom, padding, padding, left, 0,
                       cv::BORDER_CONSTANT);

    cv::Rect roi_bot(left, image.size[0] + padding, image.size[1], padding);
    cv::Rect roi_top(left, 0, image.size[1], padding);
    // blur_part(prom, roi_top, shape);
    // blur_part(prom, roi_bot, shape);
    // if (left > 0) {
    // 	cv::Rect roi_left(0, 0, left, prom.size[0]);
    // 	blur_part(prom, roi_left, shape);
    // }
    return prom;
}
