//
// SLucAM_geometry.h
//
// In this file are defined all the functions to deal with geomtry for SLAM.
//


#ifndef SLUCAM_GEOMETRY_H
#define SLUCAM_GEOMETRY_H

// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <opencv2/features2d.hpp>
#include <SLucAM_state.h>
#include <SLucAM_keypoint.h>
#include <g2o/types/sba/types_six_dof_expmap.h>



// -----------------------------------------------------------------------------
// Basic geometric functions
// -----------------------------------------------------------------------------
namespace SLucAM {
    void normalize_points(const std::vector<cv::KeyPoint>& points, \
                            std::vector<cv::KeyPoint>& normalized_points, \
                            cv::Mat& T);
    cv::Mat compute_projection_matrix(const cv::Mat& T, const cv::Mat& K);
    bool triangulate_point(const float& p1_x, const float& p1_y, \
                                    const float& p2_x, const float& p2_y, \
                                    const cv::Mat& P1, const cv::Mat& P2, \
                                    cv::Mat& p3D);
    unsigned int triangulate_points(const std::vector<cv::KeyPoint>& p_img1, \
                                    const std::vector<cv::KeyPoint>& p_img2, \
                                    const std::vector<cv::DMatch>& matches, \
                                    const std::vector<unsigned int>& idxs, \
                                    const cv::Mat& pose1, const cv::Mat& pose2, \
                                    const cv::Mat& K, \
                                    std::vector<cv::Point3f>& triangulated_points);
    void apply_perturbation_Tmatrix(const cv::Mat& perturbation, \
                                    cv::Mat& T_matrix, const unsigned int& starting_idx);
    cv::Mat invert_transformation_matrix(const cv::Mat& T_matrix);
    std::pair<int, float> nearest_3d_point(\
            const cv::Point3f& p, const std::vector<cv::Point3f>& c);
    unsigned int nearest_2d_points(const float& p_x, const float& p_y, \
                            const std::vector<cv::KeyPoint>& points, \
                            std::vector<unsigned int>& nearest_points_ids, \
                            const float threshold=100);
    float computeParallax(const cv::Mat& pose1, const cv::Mat& pose2, \
                            const std::vector<Keypoint>& keypoints, \
                            const std::vector<unsigned int>& common_landmarks_ids);
    void undistort_keypoints(const std::vector<cv::KeyPoint>& keypoints, \
                            std::vector<cv::KeyPoint>& undistorted_keypoints, \
                            const cv::Mat& distorsion_coefficients, \
                            const cv::Mat& K);
    float compute_median_distance_cam_points(const std::vector<Keypoint>& points, \
                                            const cv::Mat& pose);
    float compute_poses_distance(const cv::Mat& T1, const cv::Mat& T2);
    float compute_poses_angle(const cv::Mat& T1, const cv::Mat& T2);
    float compute_distance_2d_points(const float& p1_x, const float& p1_y, \
                                    const float& p2_x, const float& p2_y);
} // namespace SLucAM



// -----------------------------------------------------------------------------
// Representation conversion functions
// -----------------------------------------------------------------------------
namespace SLucAM {
    void quaternion_to_matrix(const cv::Mat& quaternion, cv::Mat& R);
    void matrix_to_quaternion(const cv::Mat& R, cv::Mat& quaternion);
    g2o::SE3Quat transformation_matrix_to_SE3Quat(const cv::Mat& T_matrix);
    cv::Mat SE3Quat_to_transformation_matrix(const g2o::SE3Quat& se3quat);
    Eigen::Matrix<double,3,1> point_3d_to_vector_3d(const cv::Point3f& point);
    cv::Point3f vector_3d_to_point_3d(const Eigen::Matrix<double,3,1>& vector);
    Eigen::Matrix<double,2,1> point_2d_to_vector_2d(const cv::KeyPoint& point);
} // namespace SLucAM



// -----------------------------------------------------------------------------
// Multi-view geometry functions
// -----------------------------------------------------------------------------
namespace SLucAM {
    void estimate_foundamental(const std::vector<cv::KeyPoint>& p_img1, \
                                const std::vector<cv::KeyPoint>& p_img2, \
                                const std::vector<cv::DMatch>& matches, \
                                const std::vector<unsigned int>& idxs, \
                                cv::Mat& F);
    void estimate_homography(const std::vector<cv::KeyPoint>& p_img1, \
                                const std::vector<cv::KeyPoint>& p_img2, \
                                const std::vector<cv::DMatch>& matches, \
                                const std::vector<unsigned int>& idxs, \
                                cv::Mat& H);
} // namespace SLucAM



// -----------------------------------------------------------------------------
// Projective ICP functions
// -----------------------------------------------------------------------------
namespace SLucAM {
    unsigned int perform_Posit(cv::Mat& guessed_pose, \
                                const Measurement& measurement, \
                                std::vector<bool>& points_associations_filter, \
                                const std::vector<std::pair<unsigned int, \
                                        unsigned int>>& points_associations, \
                                const std::vector<Keypoint>& keypoints, \
                                const cv::Mat& K, \
                                const float& kernel_threshold, \
                                const float& threshold_to_ignore, \
                                const unsigned int n_iterations=10, \
                                const float damping_factor=1);
    bool error_and_jacobian_Posit(const cv::Mat& guessed_pose, \
                                const cv::Point3f& guessed_landmark, \
                                const cv::KeyPoint& measured_point, \
                                const cv::Mat& K, \
                                const float& img_rows, \
                                const float& img_cols, \
                                cv::Mat& error, cv::Mat& J);
} // namespace SLucAM



#endif // SLUCAM_GEOMETRY_H