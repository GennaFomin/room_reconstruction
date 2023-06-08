#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include "boost/log/trivial.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"
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

#include "correction/borders_exposure_correction.hpp"
#include "correction/bundle_adjuster_threaded.hpp"
#include "correction/estimator.hpp"
#include "features/xfeatures2d.hpp"
#include "types/rotations.hpp"

#include "ctpl.hpp"

#ifdef PYTHON_BINDINGS
#include "boost/python.hpp"
#endif

#if ANDROID
#include "android/log.h"
#endif

#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d/nonfree.hpp"
#endif

#define ENABLE_LOG 1

#if ANDROID
#define LOGLN(msg) \
    __android_log_print(ANDROID_LOG_ERROR, "STITCHING", "%s", msg)
#else
#define LOGLN(msg) BOOST_LOG_TRIVIAL(info) << msg;
#endif

namespace stitching {
std::vector<cv::detail::MatchesInfo> match_features(
    const std::vector<cv::detail::ImageFeatures> &features,
    const float match_conf, bool is_full);

std::vector<cv::detail::CameraParams> estimate_homography(
    const std::vector<cv::detail::ImageFeatures> &features,
    const std::vector<cv::detail::MatchesInfo> &pairwise_matches,
    const std::vector<cv::Mat> &rotations);

cv::Mat stitch(std::vector<cv::Mat> images,
               std::vector<cv::detail::ImageFeatures> features,
               std::vector<cv::Mat> rotations, bool is_full);

cv::Mat stitch(std::vector<cv::Mat> images,
               std::vector<cv::detail::CameraParams> cameras);

#if defined(BACKEND) || defined(PYTHON_BINDINGS)

class Stitcher {
   private:
    int num_images;
    const std::size_t thread_pool_size = 4;
    cv::Mat full_img, img, result, result_mask;
    std::vector<cv::Mat> full_images;
    std::vector<cv::Point> corners;
    std::vector<cv::UMat> masks_warped;
    std::vector<cv::UMat> images_warped;
    std::vector<cv::UMat> images_warped_f;
    std::vector<cv::Size> sizes;
    std::vector<cv::UMat> masks;

    double work_megapix = 1.0;
    double work_scale = 1.0;
    double seam_megapix = 0.1;
    double seam_scale = 1.0;
    double seam_work_aspect = 1.0;
    double compose_scale = 1.0;
    double compose_megapix = -1;
    bool is_compose_scale_set = false;

    double default_camera_focal = 900;
    double default_camera_aspect = 1;
    double default_camera_ppx = 433;
    double default_camera_ppy = 577.5;
    double default_camera_yaw_step = 27.6;
    double default_camera_pitch_step = 27;

    float warped_image_scale;
    cv::Ptr<cv::detail::RotationWarper> warper;
    cv::Ptr<cv::WarperCreator> warper_creator =
        cv::makePtr<cv::SphericalWarper>();

    int expos_comp_type = cv::detail::ExposureCompensator::GAIN;
    int expos_comp_nr_feeds = 1;
    int expos_comp_nr_filtering = 2;
    int expos_comp_block_size = 32;
    cv::Ptr<cv::detail::ExposureCompensator> compensator =
        cv::detail::ExposureCompensator::createDefault(expos_comp_type);

    cv::Ptr<cv::detail::SeamFinder> seam_finder =
        cv::makePtr<cv::detail::DpSeamFinder>(
            cv::detail::DpSeamFinder::COLOR_GRAD);

    int blend_type = cv::detail::Blender::MULTI_BAND;
    float blend_strength = 5;

    bool try_cuda = false;

    stitching::matches_graph_t matches_graph;
    std::vector<cv::Mat> rotations;
    std::vector<cv::detail::ImageFeatures> features;
    std::vector<cv::detail::CameraParams> cameras;
    std::vector<cv::detail::MatchesInfo> matches;
    std::vector<cv::Mat> images;
    std::vector<cv::Size> full_img_sizes;

    float conf_thresh = 0.4f;
    float match_conf = 0.4f;

    bool do_wave_correct = true;
    cv::detail::WaveCorrectKind wave_correct_kind =
        cv::detail::WAVE_CORRECT_HORIZ;

    bool save_graph = false;
    std::string save_graph_to;

    std::unordered_map<int, std::string> images_;

    void init_cameras();

   public:
    bool is_full;

    Stitcher(int num_images) : num_images(num_images) {
        is_full = images.size() == static_cast<std::size_t>(39);
        init_cameras();
        images.resize(num_images);
        full_img_sizes.resize(num_images);
        corners.resize(num_images);
        masks_warped.resize(num_images);
        images_warped.resize(num_images);
        images_warped_f.resize(num_images);
        sizes.resize(num_images);
        masks.resize(num_images);
    }

    std::string version();

#ifdef PYTHON_BINDINGS
    void add_point(const boost::python::list &match);

    void add_image(std::string path, std::size_t idx);
#else
    void add_image(cv::Mat &image, std::size_t idx);
#endif
    void print_matches();

    void estimate_homography() {
        cameras = stitching::estimate_homography(features, matches, rotations);
    }

    void refine_camera_params();

    void sift();

    void compose(std::string dir) {
        scale_images();
        correct_wave();
        warp_auxiliary();
        compensate_exposure();
        find_seams();
        release();
        composite();
        flip_if_necessary();

        imwrite(dir + "/" + "result.jpg", result);
    }

#ifndef PYTHON_BINDINGS
    cv::Mat compose() { return stitching::stitch(images, cameras); }
#endif

    void scale_images();

    void correct_wave();    

    void warp_auxiliary();

    void compensate_exposure();

    void find_seams();

    void composite();

    void flip_if_necessary();

    void release();
};

BOOST_PYTHON_MODULE(stitching_core) {
    boost::python::class_<Stitcher>("Stitcher", boost::python::init<int>())
        .def("add_point", &Stitcher::add_point)
        .def("add_image", &Stitcher::add_image)
        .def("print_graph", &Stitcher::print_matches)
        .def("estimate_homography", &Stitcher::estimate_homography)
        .def("refine_cameras", &Stitcher::refine_camera_params)
        .def("sift", &Stitcher::sift)
        .def("stitch", &Stitcher::compose)
        .def("VERSION", &Stitcher::version);
}
#endif

#ifdef BACKEND
cv::Mat stitch_with_sift_from_vector(std::vector<cv::Mat> &images) {
    auto stitcher = Stitcher(images.size());
    LOGLN("Adding images");
    for (std::size_t i = 0; i < images.size(); i++)
        stitcher.add_image(images[i], i);
    stitcher.sift();
    auto result = stitcher.compose();
    if (stitcher.is_full) {
        LOGLN("Blending seam");
        result = blend_borders(result);
    }
    return result;
}
#endif

}  // namespace stitching
