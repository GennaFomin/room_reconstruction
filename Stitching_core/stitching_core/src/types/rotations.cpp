#include "./rotations.hpp"

namespace stitching {
cv::Mat cv_rotation_from_euler(double alpha, double beta, double gamma) {
    double x = alpha;
    double y = beta;
    double z = gamma;

    // Assuming the angles are in radians.
    double ch = cos(z);
    double sh = sin(z);
    double ca = cos(y);
    double sa = sin(y);
    double cb = cos(x);
    double sb = sin(x);

    double m00, m01, m02, m10, m11, m12, m20, m21, m22;

    m00 = ch * ca;
    m01 = sh * sb - ch * sa * cb;
    m02 = ch * sa * sb + sh * cb;
    m10 = sa;
    m11 = ca * cb;
    m12 = -ca * sb;
    m20 = -sh * ca;
    m21 = sh * sa * cb + ch * sb;
    m22 = -sh * sa * sb + ch * cb;

    cv::Mat rotation_matrix(3, 3, CV_64F);

    rotation_matrix.at<double>(0, 0) = m00;
    rotation_matrix.at<double>(0, 1) = m01;
    rotation_matrix.at<double>(0, 2) = m02;
    rotation_matrix.at<double>(1, 0) = m10;
    rotation_matrix.at<double>(1, 1) = m11;
    rotation_matrix.at<double>(1, 2) = m12;
    rotation_matrix.at<double>(2, 0) = m20;
    rotation_matrix.at<double>(2, 1) = m21;
    rotation_matrix.at<double>(2, 2) = m22;

    return rotation_matrix;
}

cv::Mat cv_rotation_to_euler(const cv::Mat& rotationMatrix) {
    cv::Mat euler(3, 1, CV_64F);

    double m00 = rotationMatrix.at<double>(0, 0);
    double m02 = rotationMatrix.at<double>(0, 2);
    double m10 = rotationMatrix.at<double>(1, 0);
    double m11 = rotationMatrix.at<double>(1, 1);
    double m12 = rotationMatrix.at<double>(1, 2);
    double m20 = rotationMatrix.at<double>(2, 0);
    double m22 = rotationMatrix.at<double>(2, 2);

    double x, y, z;

    // Assuming the angles are in radians.
    if (m10 > 0.998) {  // singularity at north pole
        x = 0;
        y = CV_PI / 2;
        z = atan2(m02, m22);
    } else if (m10 < -0.998) {  // singularity at south pole
        x = 0;
        y = -CV_PI / 2;
        z = atan2(m02, m22);
    } else {
        x = atan2(-m12, m11);
        y = asin(m10);
        z = atan2(-m20, m00);
    }

    euler.at<double>(0) = x;
    euler.at<double>(1) = y;
    euler.at<double>(2) = z;

    return euler;
}
}  // namespace stitching