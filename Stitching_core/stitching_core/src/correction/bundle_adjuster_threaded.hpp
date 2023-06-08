#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include "opencv2/calib3d.hpp"
#include "opencv2/calib3d/calib3d_c.h"
#include "opencv2/core/core_c.h"
#include "opencv2/core/cvdef.h"
#include "opencv2/core/eigen.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

#include <boost/iterator/counting_iterator.hpp>

#include "ceres/ceres.h"
#include "ceres/rotation.h"

#include <numeric>

#define ENABLE_LOG 1

namespace stitching {

template <typename T>
void GetCameraMat(const T *params, T *output) {
    // We manually create inverse matrix

    T f_x = params[0];
    T f_y = params[0] * params[3];
    T c_x = params[1];
    T c_y = params[2];

    output[0] = f_x;
    output[1] = T(0);
    output[2] = T(0);
    output[3] = T(0);
    output[4] = f_y;
    output[5] = T(0);
    output[6] = c_x;
    output[7] = c_y;
    output[8] = T(1);
}

struct RayError {
    RayError(const std::vector<double> p1, const std::vector<double> p2)
        : p1(p1), p2(p2) {}

    template <typename T>
    bool operator()(const T *const camera1, const T *const camera2,
                    T *residuals) const {
        T rot_mat1[9];
        T rot_mat2[9];
        T cam1_rvec[3] = {camera1[0], camera1[1], camera1[2]};
        T cam2_rvec[3] = {camera2[0], camera2[1], camera2[2]};

        ceres::EulerAnglesToRotationMatrix(camera1, 3, rot_mat1);
        ceres::EulerAnglesToRotationMatrix(camera2, 3, rot_mat2);
        Eigen::Matrix<T, 3, 3, Eigen::RowMajor> RM1 =
            Eigen::Map<const Eigen::Matrix<T, 3, 3>>(rot_mat1);
        Eigen::Matrix<T, 3, 3, Eigen::RowMajor> RM2 =
            Eigen::Map<const Eigen::Matrix<T, 3, 3>>(rot_mat2);
        RM1.transposeInPlace();
        RM2.transposeInPlace();

        T K1_inv[9];
        T K2_inv[9];
        T params1[4]{camera1[3], camera1[4], camera1[5], camera1[6]};
        T params2[4]{camera2[3], camera2[4], camera2[5], camera2[6]};
        GetCameraMat(params1, K1_inv);
        GetCameraMat(params2, K2_inv);

        Eigen::Matrix<T, 3, 3, Eigen::RowMajor> K1T =
            Eigen::Map<const Eigen::Matrix<T, 3, 3>>(K1_inv);
        Eigen::Matrix<T, 3, 3, Eigen::RowMajor> K2T =
            Eigen::Map<const Eigen::Matrix<T, 3, 3>>(K2_inv);

        Eigen::Matrix<T, 3, 3> H1 = RM1 * K1T.inverse();
        Eigen::Matrix<T, 3, 3> H2 = RM2 * K2T.inverse();
        Eigen::Matrix<T, 3, 3> H = K2T * RM2.inverse() * RM1 * K1T.inverse();

        T x1 = H(0, 0) * T(p1[0]) + H(0, 1) * T(p1[1]) + H(0, 2);
        T y1 = H(1, 0) * T(p1[0]) + H(1, 1) * T(p1[1]) + H(1, 2);
        T z1 = H(2, 0) * T(p1[0]) + H(2, 1) * T(p1[1]) + H(2, 2);

        residuals[0] = p2[0] - x1 / z1;
        residuals[1] = p2[1] - y1 / z1;

        return true;
    }

    static ceres::CostFunction *Create(const std::vector<double> p1,
                                       const std::vector<double> p2) {
        return (new ceres::AutoDiffCostFunction<RayError, 2, 7, 7>(
            new RayError(p1, p2)));
    }

    const std::vector<double> p1;
    const std::vector<double> p2;
};

using matches_graph_t =
    std::map<std::pair<int, int>,
             std::vector<std::pair<cv::Point2f, cv::Point2f>>>;

std::string RefineCameraParams(
    const std::vector<cv::detail::ImageFeatures> &features,
    const std::vector<cv::detail::MatchesInfo> &pairwise_matches,
    std::vector<cv::detail::CameraParams> &cameras, double confThresh,
    bool is_full);

std::string RefineCameraParams(matches_graph_t &match_graph,
                               std::vector<cv::detail::CameraParams> &cameras);

class BundleAdjusterCeresBase {
   public:
    BundleAdjusterCeresBase(
        const std::vector<cv::detail::ImageFeatures> &features,
        const std::vector<cv::detail::MatchesInfo> &pairwise_matches,
        std::vector<cv::detail::CameraParams> &cameras,
        double confidence_threshold, bool is_full)
        : cameras(cameras),
          confidence_threshold(confidence_threshold),
          is_full(is_full) {
        num_images = cameras.size();
        rvecs.resize(num_images);

        options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        options.preconditioner_type = ceres::SCHUR_JACOBI;
        options.use_explicit_schur_complement = true;
        options.num_threads = std::thread::hardware_concurrency();
        options.minimizer_progress_to_stdout = false;
        options.max_num_iterations = 100;

        build_graph(features, pairwise_matches);
    };

    BundleAdjusterCeresBase(std::vector<cv::detail::CameraParams> &cams,
                            matches_graph_t graph)
        : cameras(cams) {
        num_images = cameras.size();
        rvecs.resize(num_images);

        cv::SVD svd;
        for (int i = 0; i < num_images; ++i) {
            svd(cameras[i].R, cv::SVD::FULL_UV);
            cv::Mat R = svd.u * svd.vt;
            if (determinant(R) < 0) R *= -1;

            double rot[9];
            for (int k = 0; k < 3; ++k) {
                for (int j = 0; j < 3; ++j) {
                    rot[k * 3 + j] = R.at<float>(k, j);
                }
            }
            double rrvec[3], euler[3], back[9];
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

        options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        options.preconditioner_type = ceres::SCHUR_JACOBI;
        options.use_explicit_schur_complement = true;
        options.num_threads = std::thread::hardware_concurrency();
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 100;

        set_matches_graph(graph);
    }

    void run() {
        run_();
        obtain_refined_camera_params();
    };

    ceres::Solver::Summary summary;

   protected:
    virtual void run_() = 0;
    bool is_full;
    std::size_t num_images;
    std::vector<cv::detail::CameraParams> &cameras;
    std::vector<double *> rvecs;
    std::vector<std::pair<int, int>> edges;
    double confidence_threshold;

    ceres::Solver::Options options;

    std::set<int> used_cameras;
    void set_matches_graph(matches_graph_t graph) { matches_graph = graph; }
    virtual void build_graph(
        const std::vector<cv::detail::ImageFeatures> &features,
        const std::vector<cv::detail::MatchesInfo> &pairwise_matches);
    virtual void traverse_graph(const std::set<int> constant_cameras,
                                const std::set<int> variable_cameras,
                                ceres::Problem &problem, bool robustify);
    void obtain_refined_camera_params();

   private:
    matches_graph_t matches_graph;
};

class BundleAdjusterCeresLocal : public BundleAdjusterCeresBase {
    using BundleAdjusterCeresBase::BundleAdjusterCeresBase;

   protected:
    virtual void run_() override;
};

class BundleAdjusterCeresGlobal : public BundleAdjusterCeresBase {
    using BundleAdjusterCeresBase::BundleAdjusterCeresBase;

   protected:
    virtual void run_() override;
};
}  // namespace stitching
