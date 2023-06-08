#include "bundle_adjuster_threaded.hpp"
#include <thread>

using namespace std;
using namespace cv;
using namespace cv::detail;
using namespace stitching;

const auto YAW_LOWER_DEVIATION = 3;
const auto YAW_UPPER_DEVIATION = 3;
const auto PITCH_LOWER_DEVIATION = 3;
const auto PITCH_UPPER_DEVIATION = 3;
const auto ROLL_LOWER_DEVIATION = .7;
const auto ROLL_UPPER_DEVIATION = .7;
const auto FOCAL_LOWER_BOUND = 800;
const auto FOCAL_UPPER_BOUND = 1000;
const auto PPX_LOWER_DEVIATION = 10;
const auto PPX_UPPER_DEVIATION = 10;
const auto PPY_LOWER_DEVIATION = 10;
const auto PPY_UPPER_DEVIATION = 10;
const auto ASPECT_LOWER_DEVIATION = 0.15;
const auto ASPECT_UPPER_DEVIATION = 0.15;

void BundleAdjusterCeresBase::build_graph(
    const std::vector<ImageFeatures> &features,
    const std::vector<MatchesInfo> &pairwise_matches) {
    SVD svd;
    for (int i = 0; i < num_images; ++i) {
        svd(cameras[i].R, SVD::FULL_UV);
        Mat R = svd.u * svd.vt;
        if (determinant(R) < 0) R *= -1;

        double rot[9];
        for (int k = 0; k < 3; ++k) {
            for (int j = 0; j < 3; ++j) {
                rot[k * 3 + j] = R.at<float>(k, j);
            }
        }
        double euler[3];
        double pitch, roll, yaw;

        if (rot[6] * rot[6] != 1) {
            roll = asin(-rot[6]);
            pitch = atan2(rot[7] / cos(roll), rot[8] / cos(roll));
            yaw = atan2(rot[3] / cos(roll), rot[0] / cos(roll));
        } else {
            yaw = 0;
            if (rot[6] == -1) {
                roll = M_PI / 2;
                pitch = yaw + atan2(rot[1], rot[2]);
            } else {
                roll = -M_PI / 2;
                pitch = -yaw + atan2(-rot[1], -rot[2]);
            }
        }
        //
        euler[0] = pitch * 180 / M_PI;
        euler[1] = roll * 180 / M_PI;
        euler[2] = yaw * 180 / M_PI;

        auto *rvec_d = new double[7];
        rvec_d[0] = euler[0];
        rvec_d[1] = euler[1];
        rvec_d[2] = euler[2];
        rvec_d[3] = cameras[i].focal;
        rvec_d[4] = cameras[i].ppx;
        rvec_d[5] = cameras[i].ppy;
        rvec_d[6] = cameras[i].aspect;
        rvecs[i] = rvec_d;
    }

    for (int i = 0; i < num_images - 1; ++i) {
        for (int j = i + 1; j < num_images; ++j) {
            const MatchesInfo &matches_info =
                pairwise_matches[i * num_images + j];
            if (matches_info.confidence > confidence_threshold)
                edges.push_back(std::make_pair(i, j));
        }
    }

    for (const auto &edge : edges) {
        int i = edge.first;
        int j = edge.second;

        const auto &features1 = features[i];
        const auto &features2 = features[j];
        const auto &matches_info = pairwise_matches[i * num_images + j];

        for (size_t k = 0; k < matches_info.matches.size(); ++k) {
            if (!matches_info.inliers_mask[k]) continue;

            const DMatch &m = matches_info.matches[k];

            Point2f p1 = features1.keypoints[m.queryIdx].pt;
            Point2f p2 = features2.keypoints[m.trainIdx].pt;

            auto key = std::make_pair(i, j);
            matches_graph[key].push_back(std::make_pair(p1, p2));
        }
    }
}

void BundleAdjusterCeresBase::traverse_graph(
    const std::set<int> constant_cameras, const std::set<int> variable_cameras,
    ceres::Problem &problem, bool robustify) {
    for (const auto &[key, matches] : matches_graph) {
        used_cameras.clear();

        std::set<int> all_cameras;
        std::set_union(constant_cameras.begin(), constant_cameras.end(),
                       variable_cameras.begin(), variable_cameras.end(),
                       std::inserter(all_cameras, std::begin(all_cameras)));

        auto i = key.first;
        auto j = key.second;
        if (all_cameras.find(i) == all_cameras.end() ||
            all_cameras.find(j) == all_cameras.end())
            continue;

        for (const auto &pair : matches) {
            vector<double> pv1 = {static_cast<double>(pair.first.x),
                                  static_cast<double>(pair.first.y)};
            vector<double> pv2 = {static_cast<double>(pair.second.x),
                                  static_cast<double>(pair.second.y)};

            ceres::CostFunction *cost_function = RayError::Create(pv1, pv2);
            problem.AddResidualBlock(
                cost_function,
                robustify ? new ceres::SoftLOneLoss(10.0) : nullptr,
                //                                         robustify ? new
                //                                         ceres::HuberLoss(1.0)
                //                                         : nullptr, //
                //                                         SuperGlue
                rvecs[i], rvecs[j]);

            problem.SetParameterLowerBound(rvecs[i], 0,
                                           rvecs[i][0] - PITCH_LOWER_DEVIATION);
            problem.SetParameterLowerBound(rvecs[i], 1,
                                           rvecs[i][1] - ROLL_LOWER_DEVIATION);
            problem.SetParameterLowerBound(rvecs[i], 2,
                                           rvecs[i][2] - YAW_LOWER_DEVIATION);
            problem.SetParameterUpperBound(rvecs[i], 0,
                                           rvecs[i][0] + PITCH_UPPER_DEVIATION);
            problem.SetParameterUpperBound(rvecs[i], 1,
                                           rvecs[i][1] + ROLL_UPPER_DEVIATION);
            problem.SetParameterUpperBound(rvecs[i], 2,
                                           rvecs[i][2] + YAW_UPPER_DEVIATION);

            problem.SetParameterLowerBound(rvecs[j], 0,
                                           rvecs[j][0] - PITCH_LOWER_DEVIATION);
            problem.SetParameterLowerBound(rvecs[j], 1,
                                           rvecs[j][1] - ROLL_LOWER_DEVIATION);
            problem.SetParameterLowerBound(rvecs[j], 2,
                                           rvecs[j][2] - YAW_LOWER_DEVIATION);
            problem.SetParameterUpperBound(rvecs[j], 0,
                                           rvecs[j][0] + PITCH_UPPER_DEVIATION);
            problem.SetParameterUpperBound(rvecs[j], 1,
                                           rvecs[j][1] + ROLL_UPPER_DEVIATION);
            problem.SetParameterUpperBound(rvecs[j], 2,
                                           rvecs[j][2] + YAW_UPPER_DEVIATION);

            problem.SetParameterLowerBound(rvecs[i], 3, FOCAL_LOWER_BOUND);
            problem.SetParameterUpperBound(rvecs[i], 3, FOCAL_UPPER_BOUND);

            problem.SetParameterLowerBound(rvecs[j], 3, FOCAL_LOWER_BOUND);
            problem.SetParameterUpperBound(rvecs[j], 3, FOCAL_UPPER_BOUND);

            problem.SetParameterLowerBound(rvecs[i], 4,
                                           rvecs[i][4] - PPX_LOWER_DEVIATION);
            problem.SetParameterLowerBound(rvecs[i], 5,
                                           rvecs[i][5] - PPY_LOWER_DEVIATION);
            problem.SetParameterLowerBound(
                rvecs[i], 6, rvecs[i][6] - ASPECT_LOWER_DEVIATION);
            problem.SetParameterUpperBound(rvecs[i], 4,
                                           rvecs[i][4] + PPX_UPPER_DEVIATION);
            problem.SetParameterUpperBound(rvecs[i], 5,
                                           rvecs[i][5] + PPY_UPPER_DEVIATION);
            problem.SetParameterUpperBound(
                rvecs[i], 6, rvecs[i][6] + ASPECT_UPPER_DEVIATION);

            problem.SetParameterLowerBound(rvecs[j], 4,
                                           rvecs[j][4] - PPX_LOWER_DEVIATION);
            problem.SetParameterLowerBound(rvecs[j], 5,
                                           rvecs[j][5] - PPY_LOWER_DEVIATION);
            problem.SetParameterLowerBound(
                rvecs[j], 6, rvecs[j][6] - ASPECT_LOWER_DEVIATION);
            problem.SetParameterUpperBound(rvecs[j], 4,
                                           rvecs[j][4] + PPX_UPPER_DEVIATION);
            problem.SetParameterUpperBound(rvecs[j], 5,
                                           rvecs[j][5] + PPY_UPPER_DEVIATION);
            problem.SetParameterUpperBound(
                rvecs[j], 6, rvecs[j][6] + ASPECT_UPPER_DEVIATION);

            //                auto *constant_params_i = new
            //                ceres::SubsetParameterization(7, {1}); auto
            //                *constant_params_j = new
            //                ceres::SubsetParameterization(7, {1});
            //                problem.SetParameterization(rvecs[i],
            //                constant_params_i);
            //                problem.SetParameterization(rvecs[j],
            //                constant_params_j);
        }
    }

    set<int> used_constant_cameras;
    std::set_difference(used_cameras.begin(), used_cameras.end(),
                        variable_cameras.begin(), variable_cameras.end(),
                        std::inserter(used_constant_cameras,
                                      std::begin(used_constant_cameras)));

    for (const auto &idx : used_constant_cameras)
        problem.SetParameterBlockConstant(rvecs[idx]);
}

void BundleAdjusterCeresBase::obtain_refined_camera_params() {
    for (size_t i = 0; i < cameras.size(); ++i) {
        cameras[i].focal = rvecs[i][3];
        cameras[i].ppx = rvecs[i][4];
        cameras[i].ppy = rvecs[i][5];
        cameras[i].aspect = rvecs[i][6];

        double rot[9];
        ceres::EulerAnglesToRotationMatrix(rvecs[i], 3, rot);

        for (int k = 0; k < 3; ++k) {
            for (int j = 0; j < 3; ++j) {
                cameras[i].R.at<float>(k, j) = rot[k * 3 + j];
            }
        }

        Mat tmp;
        cameras[i].R.convertTo(tmp, CV_32F);
        cameras[i].R = tmp;
    }
}

// void BundleAdjusterCeresLocal::run_() {
//    auto loss_function = new ceres::SoftLOneLoss(4.0);
//    // First vertical
//    ceres::Problem problem_base;
//    std::set<int> constant_cameras{0};
//    std::set<int> variable_cameras{1, 2};
//    traverse_graph(constant_cameras, variable_cameras, loss_function,
//    problem_base); ceres::Solve(options, &problem_base, &summary); for
//    (std::size_t i = 3; i < num_images-3; i += 3) {
//        bool is_last = i == num_images - 3;
//        auto idx = static_cast<int>(i);
//        {
//            // MIDDLE
//            // [i-1] - i+2
//            //  |    \  |
//            // [i-3] - (i)
//            //  |    /  |
//            // [i-2] - i+1
//            ceres::Problem problem_middle;
//            constant_cameras = {idx - 1, idx - 2, idx - 3};
//            if (is_last && is_full) {
//                std::set<int> first_vertical{0, 1, 2};
//                constant_cameras.insert(first_vertical.begin(),
//                first_vertical.end());
//            }
//            variable_cameras = {idx};
//            traverse_graph(constant_cameras, variable_cameras, loss_function,
//            problem_middle); ceres::Solve(options, &problem_middle, &summary);
//        }
//
//        // BOTTOM
//        // i-1  -  i+2
//        //  |    x  |
//        // [i-3] - [i]
//        //  |    x  |
//        // [i-2] - (i+1)
//        ceres::Problem problem_bottom;
//        constant_cameras = {idx, idx-2, idx-3};
//        if (is_last && is_full) {
//            std::set<int> first_vertical{0, 1};
//            constant_cameras.insert(first_vertical.begin(),
//            first_vertical.end());
//        }
//        variable_cameras = {idx+1};
//        traverse_graph(constant_cameras, variable_cameras, loss_function,
//        problem_bottom); ceres::Solve(options, &problem_bottom, &summary);
//
//        // TOP
//        // [i-1] - (i+2)
//        //  |    \  |
//        // [i-3] - [i]
//        //  |    /  |
//        // i-2   -  i+1
//        ceres::Problem problem_top;
//        constant_cameras = {idx-1, idx-3, idx};
//        if (is_last && is_full) {
//            std::set<int> first_vertical{0, 2};
//            constant_cameras.insert(first_vertical.begin(),
//            first_vertical.end());
//        }
//        variable_cameras = {idx+2};
//        traverse_graph(constant_cameras, variable_cameras, loss_function,
//        problem_top); ceres::Solve(options, &problem_top, &summary);
//    }
//}

void BundleAdjusterCeresLocal::run_() {
    // First vertical
    ceres::Problem problem_base;
    std::set<int> constant_cameras{0};
    std::set<int> variable_cameras{1, 2};
    traverse_graph(constant_cameras, variable_cameras, problem_base, false);
    ceres::Solve(options, &problem_base, &summary);
    for (std::size_t i = 3; i <= num_images - 3; i += 3) {
        bool is_last = i == num_images - 3;
        auto idx = static_cast<int>(i);
        // [i-1] - (i+2)
        //  |    \  |
        // [i-3] - (i)
        //  |    /  |
        // [i-2] - (i+1)
        ceres::Problem problem_middle;
        constant_cameras = {idx - 1, idx - 2, idx - 3};
        if (is_last && is_full) {
            std::set<int> first_vertical{0, 1, 2};
            constant_cameras.insert(first_vertical.begin(),
                                    first_vertical.end());
        }
        variable_cameras = {idx, idx + 1, idx + 2};
        traverse_graph(constant_cameras, variable_cameras, problem_middle,
                       false);
        ceres::Solve(options, &problem_middle, &summary);
    }
}

void BundleAdjusterCeresGlobal::run_() {
    ceres::Problem problem;
    std::set<int> constant_cameras{0};
    std::set<int> variable_cameras(boost::counting_iterator<int>(1),
                                   boost::counting_iterator<int>(num_images));
    traverse_graph(constant_cameras, variable_cameras, problem, true);
    ceres::Solve(options, &problem, &summary);
}

string stitching::RefineCameraParams(
    const std::vector<ImageFeatures> &features,
    const std::vector<MatchesInfo> &pairwise_matches,
    std::vector<CameraParams> &cameras, double confThresh, bool is_full) {
#if 0
    auto bundle_adjuster_local = BundleAdjusterCeresLocal(features,
                                                          pairwise_matches,
                                                          cameras,
                                                          confThresh,
                                                          is_full);
    bundle_adjuster_local.run();
#endif

    auto bundle_adjuster_global = BundleAdjusterCeresGlobal(
        features, pairwise_matches, cameras, confThresh, is_full);
    bundle_adjuster_global.run();

    return bundle_adjuster_global.summary.FullReport();
}

string stitching::RefineCameraParams(matches_graph_t &matches_graph,
                                     std::vector<CameraParams> &cameras) {
    auto bundle_adjuster = BundleAdjusterCeresGlobal(cameras, matches_graph);
    bundle_adjuster.run();

    return bundle_adjuster.summary.FullReport();
};
