#include "opencv2/stitching/detail/motion_estimators.hpp"

class EstimatorCustom : public cv::detail::HomographyBasedEstimator {
   private:
    bool estimate(const std::vector<cv::detail::ImageFeatures> &features,
                  const std::vector<cv::detail::MatchesInfo> &pairwise_matches,
                  std::vector<cv::detail::CameraParams> &cameras) {
        const int num_images = static_cast<int>(features.size());

#if 0
        // Robustly estimate focal length from rotating cameras
        std::vector<Mat> Hs;
        for (int iter = 0; iter < 100; ++iter)
        {
            int len = 2 + rand()%(pairwise_matches.size() - 1);
            std::vector<int> subset;
            selectRandomSubset(len, pairwise_matches.size(), subset);
            Hs.clear();
            for (size_t i = 0; i < subset.size(); ++i)
                if (!pairwise_matches[subset[i]].H.empty())
                    Hs.push_back(pairwise_matches[subset[i]].H);
            Mat_<double> K;
            if (Hs.size() >= 2)
            {
                if (calibrateRotatingCamera(Hs, K))
                    cin.get();
            }
        }
#endif
        std::vector<double> focals;
        estimateFocal(features, pairwise_matches, focals);
        cameras.assign(num_images, cv::detail::CameraParams());
        for (int i = 0; i < num_images; ++i) cameras[i].focal = focals[i];

        // As calculations were performed under assumption that p.p. is in image
        // center
        for (int i = 0; i < num_images; ++i) {
            cameras[i].ppx += 0.5 * features[i].img_size.width;
            cameras[i].ppy += 0.5 * features[i].img_size.height;
        }

        return true;
    }
};
