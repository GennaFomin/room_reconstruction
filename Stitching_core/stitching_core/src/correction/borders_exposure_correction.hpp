#include <fstream>
#include <iostream>
#include <string>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv_modules.hpp"

// Function, that creates linear spzce between two objects
// from linear space with "elems" elements
template <typename T>
std::vector<T> linspace(T start, T end, int elems) {
    std::vector<T> res;
    for (int i = 0; i < elems; ++i) {
        res.push_back(start + (end - start) * (static_cast<double>(i) /
                                               static_cast<double>(elems)));
    }
    return res;
}

// Function, that creates mask for one channel of picture,
// dependent on correction coeffitients, calculated drom picture
cv::Mat create_channel_mask(const std::vector<double>& coefs, int h, int steps,
                            int shift) {
    // First, we compute first column of mask
    // left pic   right pic
    // _______*|_____
    // _______*|_____
    // _______*|_____
    // Pixels close to border must be changed
    // maximally   because pixels near the border
    // of the picture should be the closest
    // match to pixels from the adjacent picture
    std::vector<double> first_Col;
    for (int i = 0; i < h / steps; ++i) {
        auto scaled_Coefs = linspace(coefs[i], coefs[i + 1], steps);
        first_Col.insert(first_Col.end(), scaled_Coefs.begin(),
                         scaled_Coefs.end());
    }
    for (int i = steps * (h / steps); i < h; ++i) {
        first_Col.push_back(coefs[h / steps - 1]);
    }

    // Than we compute last column of our mask,
    // and fill it with ones, because
    // the further the pixel is, the less
    // we want to change its brightness
    std::vector<double> ones(h, 1.0);
    Eigen::VectorXd first_Col_Eig =
        Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(first_Col.data(),
                                                      first_Col.size());
    Eigen::VectorXd ones_Eig =
        Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(ones.data(), ones.size());
    // And now we fill all mask
    // example of mask
    // 1  1.5   2    2.5   3 <-- 1st coef given
    // 1  1.25	1.5  1.75  2 <-- coef from linear space
    // 1  1     1    1     1 <-- 2nd coef given
    // <--------------------
    //         linear space

    auto full_mask = linspace(ones_Eig, first_Col_Eig, shift);
    Eigen::MatrixXd mask(h, shift);
    for (int i = 0; i < shift; ++i) {
        mask.col(i) = full_mask[i];
    }
    mask.transposeInPlace();
    cv::Mat cv_Mask;
    // convert to cv Mat and return
    eigen2cv(mask, cv_Mask);

    return cv_Mask;
}

// Function, that splits image to three channels
// and counts norm of intencity for each channel
std::vector<double> split_and_count(cv::Mat& img) {
    std::vector<double> res;
    std::vector<cv::Mat> bgr;
    split(img, bgr);
    for (auto channel : bgr) {
        double nrm = norm(channel);
        res.push_back(nrm);
    }

    return res;
}

// Function, that counts coefs
// for better exposure corretion
std::vector<std::vector<double>> create_coefs(cv::Mat& img, int step) {
    int h = img.rows;
    int w = img.cols;

    std::vector<std::vector<double>> coefs(3, std::vector<double>(h / step, 1));
    // We take (step, step) squares near the right border of
    // left picture and left border of right picture, than
    // compute norms along each channel and create coefs for correction
    for (int i = 0; i < h / step; ++i) {
        cv::Rect left_Rect(w - step, i * step + 1, step, step);
        cv::Mat left_Img = img(left_Rect);
        cv::Rect right_Rect(1, i * step + 1, step, step);
        cv::Mat right_Img = img(right_Rect);

        std::vector<double> left_c = split_and_count(left_Img);
        std::vector<double> right_c = split_and_count(right_Img);

        for (int j = 0; j < 3; ++j) {
            coefs[j][i] = (right_c[j] / left_c[j]);
        }
    }

    return coefs;
}

// Function, that creates mask from
// three masks for each channel
cv::Mat create_mask(cv::Mat& img, const std::vector<std::vector<double>>& coefs,
                    int shift, int step) {
    int h = img.rows;

    auto Mask_B_Cv = create_channel_mask(coefs[0], h, step, shift);
    auto Mask_G_Cv = create_channel_mask(coefs[1], h, step, shift);
    auto Mask_R_Cv = create_channel_mask(coefs[2], h, step, shift);

    std::vector<cv::Mat> for_Merge{Mask_B_Cv, Mask_G_Cv, Mask_R_Cv};
    cv::Mat full_Mask;
    merge(for_Merge, full_Mask);

    transpose(full_Mask, full_Mask);
    full_Mask.convertTo(full_Mask, CV_32F);

    return full_Mask;
}

// Function, that perfoms
// exposure correction on borders
cv::Mat blend_borders(cv::Mat& image) {
    cv::Mat img = image.clone();

    int step = 50, shift = 500;
    int h = img.rows;
    int w = img.cols;

    auto coefs = create_coefs(img, step);
    auto mask = create_mask(img, coefs, shift, step);
    cv::Rect interest_Area(w - shift, 0, shift, h);

    cv::Mat scaled_Mat;
    cv::Mat res_Area = img(interest_Area);
    res_Area.convertTo(res_Area, CV_32F);

    cv::multiply(res_Area, mask, scaled_Mat);
    scaled_Mat.copyTo(img(interest_Area));

    return img;
}
