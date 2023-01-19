//
// SLucAM_initialization.h
//
// Description.
//


#ifndef SLUCAM_INITIALIZATION_H
#define SLUCAM_INITIALIZATION_H

// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <opencv2/features2d.hpp>
#include <SLucAM_measurement.h>
#include <SLucAM_state.h>



// -----------------------------------------------------------------------------
// Main initialization functions
// -----------------------------------------------------------------------------
namespace SLucAM {

    bool initialize(const Measurement& meas1, \
                    const Measurement& meas2, \
                    Matcher& matcher, \
                    const cv::Mat& K, cv::Mat& predicted_pose, \
                    std::vector<cv::DMatch>& matches, \
                    std::vector<unsigned int>& matches_filter, \
                    std::vector<cv::Point3f>& triangulated_points, \
                    const bool verbose=false, \
                    const unsigned int n_iters_ransac = 200);
    
    bool initialize_map(const std::vector<cv::KeyPoint>& p_img1, \
                        const std::vector<cv::KeyPoint>& p_img2, \
                        const std::vector<cv::DMatch>& matches, \
                        const std::vector<unsigned int>& matches_filter, \
                        const cv::Mat& F, const cv::Mat& K, \
                        cv::Mat& X, \
                        std::vector<cv::Point3f>& triangulated_points, \
                        unsigned int& n_inliers, \
                        const bool verbose=false, \
                        const float parallax_threshold=1.0);
    
    unsigned int compute_transformation_inliers(const std::vector<cv::KeyPoint>& p_img1, \
                                                const std::vector<cv::KeyPoint>& p_img2, \
                                                const std::vector<cv::DMatch>& matches, \
                                                const std::vector<unsigned int>& matches_filter, \
                                                const cv::Mat& R, const cv::Mat& t, \
                                                const cv::Mat& K, \
                                                std::vector<cv::Point3f>& triangulated_points, \
                                                float& parallax);

} // namespace SLucAM



// -----------------------------------------------------------------------------
// Functions to compute Essential and Homography
// -----------------------------------------------------------------------------
namespace SLucAM {
    
    const float compute_fundamental(const std::vector<cv::KeyPoint>& p_img1, \
                                    const std::vector<cv::KeyPoint>& p_img2, \
                                    const std::vector<cv::DMatch>& matches, \
                                    std::vector<unsigned int>& matches_filter, \
                                    cv::Mat& F, \
                                    const unsigned int n_iters_ransac, \
                                    const float& inliers_threshold=3.84, \
                                    const float& score_dump=5.99);
    
    const float compute_homography(const std::vector<cv::KeyPoint>& p_img1, \
                                    const std::vector<cv::KeyPoint>& p_img2, \
                                    const std::vector<cv::DMatch>& matches, \
                                    const unsigned int n_iters_ransac, \
                                    const float& inliers_threshold=5.99, \
                                    const float& score_dump=5.99);
    
    float evaluate_fundamental(const std::vector<cv::KeyPoint>& p_img1, \
                                const std::vector<cv::KeyPoint>& p_img2, \
                                const std::vector<cv::DMatch>& matches, \
                                std::vector<unsigned int>& matches_filter, \
                                const cv::Mat& F, \
                                const float& inliers_threshold, \
                                const float& score_dump);
    
    float evaluate_homography(const std::vector<cv::KeyPoint>& p_img1, \
                                const std::vector<cv::KeyPoint>& p_img2, \
                                const std::vector<cv::DMatch>& matches, \
                                std::vector<unsigned int>& matches_filter, \
                                const cv::Mat& H, \
                                const float& inliers_threshold, \
                                const float& score_dump);

} // namespace SLucAM



#endif // SLUCAM_INITIALIZATION_H