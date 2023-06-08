#include <opencv2/opencv.hpp>

constexpr double faceTransform[6][2] = {{0, 0},         {M_PI / 2, 0},
                                        {M_PI, 0},      {-M_PI / 2, 0},
                                        {0, -M_PI / 2}, {0, M_PI / 2}};

std::array<std::string, 6> faces = {"LEFT", "FRONT", "RIGHT",
                                    "BACK", "TOP",   "BOTTOM"};

void get_cubemap_face(const cv::Mat &in, cv::Mat &face, int faceId = 0,
                      const int width = -1, const int height = -1) {
    float inWidth = in.cols;
    float inHeight = in.rows;

    // Allocate map
    cv::Mat mapx(height, width, CV_32F);
    cv::Mat mapy(height, width, CV_32F);

    // Calculate adjacent (ak) and opposite (an) of the
    // triangle that is spanned from the sphere center
    // to our cube face.
    const float an = sin(M_PI / 4);
    const float ak = cos(M_PI / 4);

    const float ftu = faceTransform[faceId][0];
    const float ftv = faceTransform[faceId][1];

    // For each point in the target image,
    // calculate the corresponding source coordinates.
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Map face pixel coordinates to [-1, 1] on plane
            float nx = (float)y / (float)height - 0.5f;
            float ny = (float)x / (float)width - 0.5f;

            nx *= 2;
            ny *= 2;

            // Map [-1, 1] plane coords to [-an, an]
            // thats the coordinates in respect to a unit sphere
            // that contains our box.
            nx *= an;
            ny *= an;

            float u, v;

            // Project from plane to sphere surface.
            if (ftv == 0) {
                // Center faces
                u = atan2(nx, ak);
                v = atan2(ny * cos(u), ak);
                u += ftu;
            } else if (ftv > 0) {
                // Bottom face
                float d = sqrt(nx * nx + ny * ny);
                v = M_PI / 2 - atan2(d, ak);
                u = atan2(ny, nx);
            } else {
                // Top face
                float d = sqrt(nx * nx + ny * ny);
                v = -M_PI / 2 + atan2(d, ak);
                u = atan2(-ny, nx);
            }

            // Map from angular coordinates to [-1, 1], respectively.
            u = u / (M_PI);
            v = v / (M_PI / 2);

            // Warp around, if our coordinates are out of bounds.
            while (v < -1) {
                v += 2;
                u += 1;
            }
            while (v > 1) {
                v -= 2;
                u += 1;
            }

            while (u < -1) {
                u += 2;
            }
            while (u > 1) {
                u -= 2;
            }

            // Map from [-1, 1] to in texture space
            u = u / 2.0f + 0.5f;
            v = v / 2.0f + 0.5f;

            u = u * (inWidth - 1);
            v = v * (inHeight - 1);

            // Save the result for this pixel in map
            mapx.at<float>(x, y) = u;
            mapy.at<float>(x, y) = v;
        }
    }

    // Recreate output image if it has wrong size or type.
    if (face.cols != width || face.rows != height || face.type() != in.type()) {
        face = cv::Mat(width, height, in.type());
    }

    // Do actual resampling using OpenCV's remap
    cv::remap(in, face, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_CONSTANT,
              cv::Scalar(0, 0, 0));
}

std::unordered_map<std::string, cv::Mat> equirectangular_to_cubemap(
    cv::Mat panorama) {
    std::unordered_map<std::string, cv::Mat> result;

    for (std::size_t i = 0; i < 6; i++) {
        cv::Mat face;
        get_cubemap_face(panorama, face, i, 2000, 2000);
        if (i > 3) cv::rotate(face, face, cv::ROTATE_180);
        result[faces[i]] = face;
    }

    return result;
}
