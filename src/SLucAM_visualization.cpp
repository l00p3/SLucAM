//
// SLucAM_visualization.h implementation
//


// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <SLucAM_visualization.h>
#include <SLucAM_geometry.h>
#include <iostream>



// -----------------------------------------------------------------------------
// Implementation of utilities to visualize data on terminal
// -----------------------------------------------------------------------------
namespace SLucAM {
    
    /*
    * This function allows to visualize a pose (given as a transformation 
    * matrix) in the following way:
    *   <position.x [m], position.y [m], position.z [m], quaternion.w [], 
    *    quaternion.x [], qquaternion.y [], quaternion.z []>
    */
    void visualize_pose_as_quaternion(const cv::Mat& pose) {

        // Take the rotational part
        cv::Mat R = pose(cv::Rect(0,0,3,3));

        // Transform it in quaternion representation
        cv::Mat quaternion;
        SLucAM::matrix_to_quaternion(R, quaternion);

        // Visualize
        std::cout <<
            pose.at<float>(0,3) << " " << \
            pose.at<float>(1,3) << " " << \
            pose.at<float>(2,3) << " " << \
            quaternion.at<float>(0,0) << " " << \
            quaternion.at<float>(1,0) << " " << \
            quaternion.at<float>(2,0) << " " << \
            quaternion.at<float>(3,0) 
        << std::endl;

    }
} // namespace SLucAM