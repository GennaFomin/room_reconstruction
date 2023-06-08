project(stitching_core)
cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "release")

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS log_setup log)
include_directories(${OpenCV_INCLUDE_DIRS}) #1
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${Eigen3_INCLUDE_DIRS})


add_library(stitching_core MODULE
	/src/correction/bundle_adjuster_threaded.hpp
	/src/correction/bundle_adjuster_threaded.cpp
	/src/features/xfeatures2d.hpp
	/src/features/sift.cpp
	/src/types/rotations.cpp
	/src/types/rotations.cpp
	/src/pipeline.hpp
	/src/pipeline.cpp
	/src/ctpl.hpp
)

set_target_properties(stitching_core PROPERTIES PREFIX "")

execute_process (
    COMMAND bash -c "python3 -c \"from sysconfig import get_paths as gp; print(gp()['include'])\""
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
target_include_directories(stitching_core PRIVATE ${PYTHON_INCLUDE_DIR})
target_link_libraries(stitching_core "/boost/stage/lib/libboost_python36-3.so.1.73.0")
target_link_libraries(stitching_core
	${Boost_LIBRARIES}
	${OpenCV_LIBRARIES}
	${CERES_LIBRARIES}
	${Eigen3_LIBRARIES}
)
