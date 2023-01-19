//
// SLucAM_visualization.h
//
// In this module we have all the function to visualize data user-friendly
//


#ifndef SLUCAM_VISUALIZATION_H
#define SLUCAM_VISUALIZATION_H

// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <opencv2/features2d.hpp>



// -----------------------------------------------------------------------------
// Utilities to visualize data on terminal
// -----------------------------------------------------------------------------
namespace SLucAM {
    void visualize_pose_as_quaternion(const cv::Mat& T);
} // namespace SLucAM



#endif // SLUCAM_VISUALIZATION_H