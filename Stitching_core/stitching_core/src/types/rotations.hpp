#pragma once

#include "opencv2/opencv.hpp"

namespace stitching {
cv::Mat cv_rotation_from_euler(double alpha, double beta, double gamma);

cv::Mat cv_rotation_to_euler(cv::Mat);
}  // namespace stitching