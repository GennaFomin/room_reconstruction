#include "opencv2/opencv.hpp"
#include "opencv2/opencv_modules.hpp"

// We find first row from chosen side that has very a few nonzero elements
// We use binary search for better performance
// General function for binary search
template <typename Func>
int binary_nonzero(const cv::Mat &image, Func func, int axis, int left,
                   int right) {
    while (right - left > 1) {
        int middle = (right + left) / 2;
        cv::Mat row;
        if (axis == 0) {
            row = image.row(middle);
        } else {
            row = image.col(middle);
        }
        row.convertTo(row, CV_32F);
        cv::cvtColor(row, row, cv::COLOR_BGR2GRAY);
        int count_nonzero = cv::countNonZero(row);
        if (func(row.size[1 - axis], count_nonzero)) {
            right = middle;
        } else {
            left = middle;
        }
    }
    return right;
}

// Also we use ternary search for good performance
template <typename Func>
int ternary_nonzero(const cv::Mat &image, Func func, int axis, int left,
                    int right) {
    while (right - left > 3) {
        int l_mid = (right + 2 * left) / 3;
        int r_mid = (2 * right + left) / 3;
        cv::Mat row_l, row_r;
        if (axis == 0) {
            row_l = image.row(l_mid);
            row_r = image.row(r_mid);
        } else {
            row_l = image.col(l_mid);
            row_r = image.col(r_mid);
        }
        row_l.convertTo(row_l, CV_32F);
        cv::cvtColor(row_l, row_l, cv::COLOR_BGR2GRAY);
        row_r.convertTo(row_r, CV_32F);
        cv::cvtColor(row_r, row_r, cv::COLOR_BGR2GRAY);
        int count_nonzero_l = cv::countNonZero(row_l);
        int count_nonzero_r = cv::countNonZero(row_r);
        if (func(count_nonzero_l, count_nonzero_r)) {
            right = r_mid;
        } else {
            left = l_mid;
        }
    }
    return left;
}

// Function for finding top border
int find_nonzero_top(const cv::Mat &pano) {
    int top = ternary_nonzero(
        pano, [](int first, int second) { return first >= second; }, 0, 0,
        pano.size[0] / 3);
    return top;
}

// Function for finding bot border
int find_nonzero_bot(const cv::Mat &pano) {
    int bot = ternary_nonzero(
        pano, [](int first, int second) { return first > second; }, 0,
        2 * pano.size[0] / 3, pano.size[0]);
    return bot;
}

// Function for finding left border
// !!! Use only after top and bot cutting !!!
int find_nonzero_left(const cv::Mat &pano) {
    int left = binary_nonzero(
        pano, [](int first, int second) { return first - second <= 10; }, 1, 0,
        pano.size[1] / 3);
    return left;
}

// Function for finding right border
// !!! Use only after top and bot cutting !!!
int find_nonzero_right(const cv::Mat &pano) {
    int right = binary_nonzero(
        pano, [](int first, int second) { return first - second >= 10; }, 1,
        2 * pano.size[1] / 3, pano.size[1]);
    return right;
}
