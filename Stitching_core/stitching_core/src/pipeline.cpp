#include "pipeline.hpp"
#include <future>
#include <numeric>

using namespace cv;
using namespace cv::detail;
using namespace std;

#ifdef PYTHON_BINDINGS
template <typename T>
T pyget(const boost::python::list &list, std::size_t idx) {
    return boost::python::extract<T>(list[idx]);
}
#endif

namespace stitching {
std::string Stitcher::version() { return "20.26"; }

void Stitcher::init_cameras() {
    int r = 0;
    for (std::size_t i = 0; i < num_images; i += 3) {
        cv::detail::CameraParams camera;
        cv::Mat R;

        R = cv_rotation_from_euler(0, 0, r * 27.6 * M_PI / 180);
        rotations.push_back(R);
        camera.R = R;
        cameras.push_back(camera);

        R = cv_rotation_from_euler(-default_camera_pitch_step * M_PI / 180, 0,
                                   r * default_camera_yaw_step * M_PI / 180);
        rotations.push_back(R);
        camera.R = R;
        cameras.push_back(camera);

        R = cv_rotation_from_euler(default_camera_pitch_step * M_PI / 180, 0,
                                   r * default_camera_yaw_step * M_PI / 180);
        rotations.push_back(R);
        camera.R = R;
        cameras.push_back(camera);

        r++;
    }

    for (size_t i = 0; i < cameras.size(); ++i)
        cameras[i].R.convertTo(cameras[i].R, CV_32F);
}

void Stitcher::add_point(const boost::python::list &match) {
    int left_idx = pyget<int>(match, 0);
    int right_idx = pyget<int>(match, 1);
    cv::Point2f left_point(pyget<float>(match, 2), pyget<float>(match, 3));
    cv::Point2f right_point(pyget<float>(match, 4), pyget<float>(match, 5));
    if (left_idx > right_idx) {
        std::swap(left_idx, right_idx);
        std::swap(left_point, right_point);
    }
    auto vertices = std::make_pair(left_idx, right_idx);
    matches_graph[vertices].push_back(std::make_pair(left_point, right_point));
}

void Stitcher::print_matches() {
    for (const auto &[vertices, matches] : matches_graph) {
        auto key = std::to_string(vertices.first) + "-" +
                   std::to_string(vertices.second);
        std::cout << key << ": " << matches.size() << std::endl;
    }
}

#ifdef PYTHON_BINDINGS
void Stitcher::add_image(std::string path, std::size_t idx) {
    const auto it = images_.find(idx);
    if (it != images_.end() && it->second == path) return;
    LOGLN(("Reading image " + path + " # " + std::to_string(idx)).c_str());
    images_[idx] = path;
    images[idx] = cv::imread(path);
}
#else
void add_image(cv::Mat &image, std::size_t idx) { images[idx] = image; }
#endif

void Stitcher::refine_camera_params() {
    auto report = stitching::RefineCameraParams(features, matches, cameras,
                                                conf_thresh, is_full);
    LOGLN(report);
}

void Stitcher::sift() {
    LOGLN("Extracting features");
    auto feature_extractor = cv::xfeatures2d::SIFT::create();
    for (std::size_t i = 0; i < images.size(); ++i) {
        cv::Mat original = images[i];
        auto work_scale = sqrt(1e6 / original.size().area());
        cv::Mat preprocessed;
        cv::cvtColor(original, preprocessed, cv::COLOR_RGB2GRAY);
        auto clahe = cv::createCLAHE();
        clahe->setClipLimit(8);
        clahe->apply(preprocessed, preprocessed);
        resize(preprocessed, preprocessed, cv::Size(), work_scale, work_scale,
               cv::INTER_LINEAR_EXACT);
        cv::detail::ImageFeatures result;
        cv::detail::computeImageFeatures(feature_extractor, preprocessed,
                                         result);
        // RootSIFT
        for (int i = 0; i < result.descriptors.rows; ++i)
            cv::normalize(result.descriptors.row(i), result.descriptors.row(i),
                          1.0, 0.0, cv::NORM_L1);
        cv::sqrt(result.descriptors, result.descriptors);
        result.img_idx = i;
        features.push_back(result);
    }

    matches = stitching::match_features(features, match_conf, is_full);
}

void Stitcher::scale_images() {
    num_images = images.size();
    bool is_work_scale_set = false, is_seam_scale_set = false;

    for (int i = 0; i < num_images; ++i) {
        full_images.push_back(images[i]);
        full_img = images[i];
        full_img_sizes[i] = full_img.size();

        if (work_megapix < 0) {
            img = full_img;
            work_scale = 1;
            is_work_scale_set = true;
        } else {
            if (!is_work_scale_set) {
                work_scale =
                    min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
                is_work_scale_set = true;
            }
            resize(full_img, img, Size(), work_scale, work_scale,
                   INTER_LINEAR_EXACT);
        }
        if (!is_seam_scale_set) {
            seam_scale =
                min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
            seam_work_aspect = seam_scale / work_scale;
            is_seam_scale_set = true;
        }

        resize(full_img, img, Size(), seam_scale, seam_scale,
               INTER_LINEAR_EXACT);
        images[i] = img.clone();
    }

    full_img.release();
    img.release();
}

void Stitcher::correct_wave() {
    vector<Mat> rmats;
    for (size_t i = 0; i < cameras.size(); ++i)
        rmats.push_back(cameras[i].R.clone());
    waveCorrect(rmats, wave_correct_kind);
    for (size_t i = 0; i < cameras.size(); ++i) cameras[i].R = rmats[i];
}

void Stitcher::warp_auxiliary() {
    LOGLN("Warping images (auxiliary)");

    // Prepare images masks
    for (int i = 0; i < num_images; ++i) {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    // Find median focal length
    vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i)
        focals.push_back(cameras[i].focal);

    sort(focals.begin(), focals.end());
    if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] +
                                                focals[focals.size() / 2]) *
                             0.5f;

    // Warp images and their masks
    ctpl::thread_pool pool(thread_pool_size);
    std::vector<std::future<void>> init_warps_futures(num_images);
    auto swa = static_cast<float>(seam_work_aspect);
    for (int i = 0; i < num_images; ++i) {
        auto future = pool.push([&, swa, i](int) {
            auto warper = warper_creator->create(
                static_cast<float>(warped_image_scale * swa));

            Mat_<float> K;
            cameras[i].K().convertTo(K, CV_32F);
            K(0, 0) *= swa;
            K(0, 2) *= swa;
            K(1, 1) *= swa;
            K(1, 2) *= swa;

            corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR,
                                      BORDER_REFLECT, images_warped[i]);
            sizes[i] = images_warped[i].size();
            warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST,
                         BORDER_CONSTANT, masks_warped[i]);
            images_warped[i].convertTo(images_warped_f[i], CV_32F);
        });
        init_warps_futures[i] = std::move(future);
    }
    for (int i = 0; i < num_images; ++i) init_warps_futures[i].get();
}

void Stitcher::compensate_exposure() {
    GainCompensator *gcompensator =
        dynamic_cast<GainCompensator *>(compensator.get());
    gcompensator->setNrFeeds(expos_comp_nr_feeds);
    compensator->feed(corners, images_warped, masks_warped);
}

void Stitcher::find_seams() {
    seam_finder->find(images_warped_f, corners, masks_warped);
}

void Stitcher::composite() {
    Ptr<Blender> blender;
    double compose_work_aspect = 1;

    if (!is_compose_scale_set) {
        if (compose_megapix > 0)
            compose_scale =
                min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
        is_compose_scale_set = true;

        // Compute relative scales
        compose_work_aspect = compose_scale / work_scale;

        // Update warped image scale
        warped_image_scale *= static_cast<float>(compose_work_aspect);
        warper = warper_creator->create(warped_image_scale);

        // Update corners and sizes
        for (int i = 0; i < num_images; ++i) {
            // Update intrinsics
            cameras[i].focal *= compose_work_aspect;
            cameras[i].ppx *= compose_work_aspect;
            cameras[i].ppy *= compose_work_aspect;

            // Update corner and size
            Size sz = full_img_sizes[i];
            if (std::abs(compose_scale - 1) > 1e-1) {
                sz.width = cvRound(full_img_sizes[i].width * compose_scale);
                sz.height = cvRound(full_img_sizes[i].height * compose_scale);
            }

            Mat K;
            cameras[i].K().convertTo(K, CV_32F);
            Rect roi = warper->warpRoi(sz, K, cameras[i].R);
            corners[i] = roi.tl();
            sizes[i] = roi.size();
        }
    }

    blender = Blender::createDefault(blend_type, try_cuda);
    Size dst_sz = resultRoi(corners, sizes).size();
    float blend_width =
        sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
    MultiBandBlender *mb = dynamic_cast<MultiBandBlender *>(blender.get());
    mb->setNumBands(
        static_cast<int>(ceil(std::log(blend_width) / std::log(2.)) - 1.));
    blender->prepare(corners, sizes);

    ctpl::thread_pool pool(thread_pool_size);
    std::vector<std::future<void>> compositing_futures(num_images);
    for (int img_idx = 0; img_idx < num_images; ++img_idx) {
        auto task = pool.push([&, img_idx](int) {
            Mat f_img, im;
            Mat img_warped, img_warped_s;
            Mat dilated_mask, seam_mask, mask, mask_warped;

            auto Warper = warper_creator->create(warped_image_scale);
            // Read image and resize it if necessary
            f_img = full_images[img_idx];

            if (abs(compose_scale - 1) > 1e-1)
                resize(full_img, img, Size(), compose_scale, compose_scale,
                       INTER_LINEAR_EXACT);
            else
                im = f_img;
            f_img.release();
            Size img_size = im.size();

            Mat K;
            cameras[img_idx].K().convertTo(K, CV_32F);

            // Warp the current image
            Warper->warp(im, K, cameras[img_idx].R, INTER_LINEAR,
                         BORDER_REFLECT, img_warped);

            // Warp the current image mask
            mask.create(img_size, CV_8U);
            mask.setTo(Scalar::all(255));
            Warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST,
                         BORDER_CONSTANT, mask_warped);

            // Compensate exposure
            compensator->apply(img_idx, corners[img_idx], img_warped,
                               mask_warped);

            img_warped.convertTo(img_warped_s, CV_16S);
            img_warped.release();
            img.release();
            mask.release();

            dilate(masks_warped[img_idx], dilated_mask, Mat());
            resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0,
                   INTER_LINEAR_EXACT);
            mask_warped = seam_mask & mask_warped;

            blender->feed(img_warped_s, mask_warped, corners[img_idx]);
        });
        compositing_futures[img_idx] = std::move(task);
    }

    for (size_t i = 0; i < num_images; ++i) compositing_futures[i].get();

    // Mat result, result_mask;
    blender->blend(result, result_mask);
}

void Stitcher::flip_if_necessary() {
    if (corners[0].y > corners[1].y) flip(result, result, -1);
}

void Stitcher::release() {
    images.clear();
    images_warped.clear();
    images_warped_f.clear();
    masks.clear();
}

Mat stitch_(std::vector<Mat> &images, std::vector<CameraParams> &cameras) {}

std::vector<MatchesInfo> match_features(
    const std::vector<ImageFeatures> &features, const float match_conf,
    bool is_full) {
    LOGLN("Pairwise matching");
#if ENABLE_LOG
    auto t = getTickCount();
#endif
    const auto num_images = static_cast<int>(features.size());
    vector<MatchesInfo> pairwise_matches;
    Ptr<FeaturesMatcher> matcher;
    matcher = makePtr<BestOf2NearestMatcher>(false, match_conf);

    Mat match_mask(features.size(), features.size(), CV_8U, Scalar(0));
    // 2 - 5 - 8 - ...    Matcher expects images to be sorted in the following
    // manner: | \ | \ | \        MIDDLE -> BOTTOM -> TOP -> MIDDLE -> BOTTOM ->
    // ... 0 - 3 - 6 - ... | / | / | / 1 - 4 - 7 - ...
    for (int i = 0; i < num_images; i += 3) {
        match_mask.at<char>(i, i + 1) = 1;
        match_mask.at<char>(i, i + 2) = 1;
        if ((i + 3) < num_images) match_mask.at<char>(i, i + 3) = 1;
        if (i > 0) {
            match_mask.at<char>(i - 1, i) = 1;
            match_mask.at<char>(i - 2, i) = 1;
            if ((i + 1) < num_images) {
                match_mask.at<char>(i - 2, i + 1) = 1;
                match_mask.at<char>(i - 3, i + 1) = 1;
            }
            if ((i + 2) < num_images) {
                match_mask.at<char>(i - 1, i + 2) = 1;
                match_mask.at<char>(i - 3, i + 2) = 1;
            }
        }
    }
    if (is_full) {
        match_mask.at<char>(0, num_images - 3) = 1;
        match_mask.at<char>(0, num_images - 2) = 1;
        match_mask.at<char>(0, num_images - 1) = 1;
        match_mask.at<char>(1, num_images - 3) = 1;
        match_mask.at<char>(1, num_images - 2) = 1;
        match_mask.at<char>(2, num_images - 2) = 1;
        match_mask.at<char>(2, num_images - 1) = 1;
    }

    auto um = match_mask.getUMat(ACCESS_READ);
    (*matcher)(features, pairwise_matches, um);
    matcher->collectGarbage();

    return pairwise_matches;
}

vector<CameraParams> estimate_homography(
    const vector<ImageFeatures> &features,
    const vector<MatchesInfo> &pairwise_matches, const vector<Mat> &rotations) {
    Ptr<Estimator> estimator = makePtr<EstimatorCustom>();
    vector<CameraParams> cameras(features.size());
    LOGLN("Homography estimation");
    if (!(*estimator)(features, pairwise_matches, cameras))
        cout << "Homography estimation failed.\n";

    for (size_t i = 0; i < cameras.size(); ++i)
        rotations[i].convertTo(cameras[i].R, CV_32F);

    return cameras;
}

cv::Mat stitch(std::vector<cv::Mat> images,
               std::vector<cv::detail::ImageFeatures> features,
               std::vector<cv::Mat> rotations, bool is_full) {
    auto matches = match_features(features, .4, is_full);
    auto cameras = estimate_homography(features, matches, rotations);
    auto result = stitch_(images, cameras);

    return result;
}

cv::Mat stitch(std::vector<cv::Mat> images, std::vector<CameraParams> cameras) {
    auto result = stitch_(images, cameras);

    return result;
}
}  // namespace stitching
