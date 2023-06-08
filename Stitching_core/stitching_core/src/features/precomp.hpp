#ifndef __OPENCV_PRECOMP_H__
#define __OPENCV_PRECOMP_H__

//#include "opencv2/xfeatures2d/cuda.hpp"

#include "opencv2/imgproc.hpp"
#include "xfeatures2d.hpp"

//#include "opencv2/core/utility.hpp"
//#include "opencv2/core/private.hpp"
//#include "opencv2/core/private.cuda.hpp"
#include "opencv2/core/ocl.hpp"

#include "opencv2/opencv_modules.hpp"

#ifdef HAVE_OPENCV_CUDAARITHM
#include "opencv2/cudaarithm.hpp"
#endif

//#include "opencv2/core/private.hpp"

#define USE_AVX2 (cv::checkHardwareSupport(CV_CPU_AVX2))

#endif