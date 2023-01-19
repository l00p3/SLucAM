//
// SLucAM_geometry.h implementation
//


// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <SLucAM_geometry.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <limits>
#include <Eigen/Eigen>
#include <Eigen/SVD>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

// TODO delete this
#include <iostream>
using namespace std;



// -----------------------------------------------------------------------------
// Implementation of basic geometric functions
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * A simple normalization function for points. It shifts all points and bring
    * their centroid to the origin and then scale them in such a way their 
    * average distance from the origin is sqrt(2).
    * Inputs:
    *   points: points to normalize
    *   normalized_points: output points, normalized
    *   T: transformation matrix to normalize the point (useful
    *       for denormalization)
    */
    void normalize_points(const std::vector<cv::KeyPoint>& points, \
                            std::vector<cv::KeyPoint>& normalized_points, \
                            cv::Mat& T) {
        
        // Initialization
        unsigned int n_points = points.size();
        normalized_points.resize(n_points);
        T = cv::Mat::eye(3, 3, CV_32F);
        
        // Compute the centroid of the points
        float mu_x = 0;
        float mu_y = 0;
        for(const cv::KeyPoint& p: points) {
            mu_x += p.pt.x;
            mu_y += p.pt.y;
        }
        mu_x /= n_points;
        mu_y /= n_points;

        // Shift the points such that the centroid will be the origin
        // and in the meantime compute the average distance from the
        // origin
        float average_distance_x = 0;
        float average_distance_y = 0;
        for(unsigned int i=0; i<n_points; i++) {
            normalized_points[i].pt.x = points[i].pt.x-mu_x;
            normalized_points[i].pt.y = points[i].pt.y-mu_y;
            
            average_distance_x += std::fabs(normalized_points[i].pt.x);
            average_distance_y += std::fabs(normalized_points[i].pt.y);
        }
        average_distance_x /= n_points;
        average_distance_y /= n_points;
        
        // Scale the points such that the average distance from 
        // the origin is 1
        float scale_x = 1.0/average_distance_x;
        float scale_y = 1.0/average_distance_y;
        for(unsigned int i=0; i<n_points; i++) {
            normalized_points[i].pt.x *= scale_x;
            normalized_points[i].pt.y *= scale_y;
        }

        // Ensemble the T matrix
        T = cv::Mat::eye(3, 3, CV_32F);
        T.at<float>(0,0) = scale_x;
        T.at<float>(1,1) = scale_y;
        T.at<float>(0,2) = -mu_x*scale_x;
        T.at<float>(1,2) = -mu_y*scale_y; 
    }



    /*
    * Function that, givena transformation matrix (T) and a camera matrix (K)
    * returns the 3x4 projection matrix P=K*[I|0]*T
    */
    cv::Mat compute_projection_matrix(const cv::Mat& T, const cv::Mat& K) {
        const cv::Mat R = T.rowRange(0,3).colRange(0,3);
        const cv::Mat t = T.rowRange(0,3).col(3);
        cv::Mat P(3,4,CV_32F);
        R.copyTo(P.rowRange(0,3).colRange(0,3));
        t.copyTo(P.rowRange(0,3).col(3));
        return K*P;
    }


    /*
    * Function that given a point seen from camera 1 (p1_x, p1_y) and the same
    * point seen from the camera 2 (p2_x, p2_y) and the corresponding projection
    * matrices (P1 and P2), compute the corresponding 3D point by linear triangulation.
    */
    bool triangulate_point(const float& p1_x, const float& p1_y, \
                                    const float& p2_x, const float& p2_y, \
                                    const cv::Mat& P1, const cv::Mat& P2, \
                                    cv::Mat& p3D) {
        // Ensemble the A matrix of equations
        cv::Mat A(4,4,CV_32F);
        A.row(0) = p1_x*P1.row(2)-P1.row(0);
        A.row(1) = p1_y*P1.row(2)-P1.row(1);
        A.row(2) = p2_x*P2.row(2)-P2.row(0);
        A.row(3) = p2_y*P2.row(2)-P2.row(1);

        // Go in the Eigen representation
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, \
                                    Eigen::Dynamic, \
                                    Eigen::RowMajor>> A_Eigen \
                                    (A.ptr<float>(), A.rows, A.cols);

        // Triangulate with linear method
        Eigen::JacobiSVD<Eigen::Matrix4f> svd_A(A_Eigen, Eigen::ComputeFullV);
        Eigen::Vector4f point_3D_eigen = svd_A.matrixV().col(3);

        // Check the validity of the triangulated point
        if(point_3D_eigen(3)==0)
            return false;

        // Put it in Euclidean coordinates
        Eigen::Vector3f point_3D_new = point_3D_eigen.head(3)/point_3D_eigen(3);

        // Check if the point is at a finite position
        if(!isfinite(point_3D_new(0)) || \
            !isfinite(point_3D_new(1)) || \
            !isfinite(point_3D_new(2))) {
            return false;
        }
        
        // Back to OpenCV representation
        cv::eigen2cv(point_3D_new, p3D);

        return true;

    }



    /*
    * Function that triangulates a bunch of points seen from two cameras.
    * Inputs:
    *   p_img1/p_img2: points seen from two cameras
    *   matches: all matches between p_img1 and p_img2, with outliers
    *   idxs: we will consider only those points contained in 
    *           the matches vector, for wich we have indices in this vector idxs
    *   pose1/pose2: pose of the two cameras (world wrt camera)
    *   K: camera matrix of the two cameras
    *   triangulated_points: vector where to store the triangulated points
    *       (expressed w.r.t. world)
    * Outputs:
    *   n_triangulated_points: number of triangulated points
    */
    unsigned int triangulate_points(const std::vector<cv::KeyPoint>& p_img1, \
                                    const std::vector<cv::KeyPoint>& p_img2, \
                                    const std::vector<cv::DMatch>& matches, \
                                    const std::vector<unsigned int>& idxs, \
                                    const cv::Mat& pose1, const cv::Mat& pose2, \
                                    const cv::Mat& K, \
                                    std::vector<cv::Point3f>& triangulated_points) {
        
        // Initialization
        const unsigned int n_points = idxs.size();
        unsigned int n_triangulated_points = 0;
        const cv::Mat P1 = compute_projection_matrix(pose1, K);
        const cv::Mat P2 = compute_projection_matrix(pose2, K);
        float current_cos_parallax, imx, imy, invz;
        cv::Mat d1, d2;
        const float reprojection_threshold = 4;
        const float& fx = K.at<float>(0,0);
        const float& fy = K.at<float>(1,1);
        const float& cx = K.at<float>(0,2);
        const float& cy = K.at<float>(1,2);

        // Compute the origins of the two cameras
        const cv::Mat& R1 = pose1.rowRange(0,3).colRange(0,3);
        const cv::Mat& R2 = pose2.rowRange(0,3).colRange(0,3);
        const cv::Mat& t1 = pose1.rowRange(0,3).col(3);
        const cv::Mat& t2 = pose2.rowRange(0,3).col(3);
        const cv::Mat O1 = -R1.t()*t1;
        const cv::Mat O2 = -R2.t()*t2;
        
        // Triangulate each couple of points
        triangulated_points.reserve(n_points);
        for(unsigned int i=0; i<n_points; ++i) {

            // Insert an invalid point (if it passes all the checks it
            // wil be updated), an invalid point is assumed at position (0,0,0)
            // TODO: this assumption can be avoided
            triangulated_points.emplace_back(cv::Point3f(0,0,0));

            // Take references to the current couple of points
            const float& p1_x = p_img1[matches[idxs[i]].queryIdx].pt.x;
            const float& p1_y = p_img1[matches[idxs[i]].queryIdx].pt.y;
            const float& p2_x = p_img2[matches[idxs[i]].trainIdx].pt.x;
            const float& p2_y = p_img2[matches[idxs[i]].trainIdx].pt.y;

            // Triangulate with the linear method
            cv::Mat p3D;
            if(!triangulate_point(p1_x, p1_y, p2_x, p2_y, P1, P2, p3D)) {
                continue;
            }
            const float& p3D_x = p3D.at<float>(0);
            const float& p3D_y = p3D.at<float>(1);
            const float& p3D_z = p3D.at<float>(2);

            // Check the parallax
            d1 = p3D - O1;
            d2 = p3D - O2;
            current_cos_parallax = d1.dot(d2)/(cv::norm(d1)*cv::norm(d2));
            if(current_cos_parallax<0 || current_cos_parallax>0.99998) {
                continue;
            }

            // Check if the point is in front of the first camera
            cv::Mat p3D_wrt1 = R1*p3D+t1;
            const float& p3D_wrt1_x = p3D_wrt1.at<float>(0);
            const float& p3D_wrt1_y = p3D_wrt1.at<float>(1);
            const float& p3D_wrt1_z = p3D_wrt1.at<float>(2);
            if(p3D_wrt1_z<=0)
                continue;
            
            // Check if the point is in front of the second camera
            cv::Mat p3D_wrt2 = R2*p3D+t2;
            const float& p3D_wrt2_x = p3D_wrt2.at<float>(0);
            const float& p3D_wrt2_y = p3D_wrt2.at<float>(1);
            const float& p3D_wrt2_z = p3D_wrt2.at<float>(2);
            if(p3D_wrt2_z<=0)
                continue;  

            // Check reprojection error in first image
            invz = 1.0/p3D_wrt1_z;
            imx = fx*p3D_wrt1_x*invz+cx;
            imy = fy*p3D_wrt1_y*invz+cy;
            if(((imx-p1_x)*(imx-p1_x)+(imy-p1_y)*(imy-p1_y)) > \
                    reprojection_threshold)
                continue;

            // Check reprojection error in second image
            invz = 1.0/p3D_wrt2_z;
            imx = fx*p3D_wrt2_x*invz+cx;
            imy = fy*p3D_wrt2_y*invz+cy;
            if(((imx-p2_x)*(imx-p2_x)+(imy-p2_y)*(imy-p2_y)) > \
                    reprojection_threshold)
                continue;   

            // It passes all the checks, so it is valid and can be saved
            triangulated_points.back().x = p3D_x;
            triangulated_points.back().y = p3D_y;
            triangulated_points.back().z = p3D_z;

            // Count it as good
            ++n_triangulated_points;

        }
        triangulated_points.shrink_to_fit();

        return n_triangulated_points;
    }



    /*
    * Function that, given a transformation matrix and a perturbation vector
    * where from position "starting_idx" to "starting_idx+6" contains the
    * [tx, ty, tz, x-angle, y-angle, z-angle] perturbation to apply.
    * Inputs:
    *   perturbation: the perturbation vector
    *   T_matrix: the transformation matrix to "perturb"
    *   starting_idx: the position from which we have the updates for T_matrix 
    */
    void apply_perturbation_Tmatrix(const cv::Mat& perturbation, \
                                    cv::Mat& T_matrix, const unsigned int& starting_idx) {
        
        // Some reference to save time
        const float& tx = perturbation.at<float>(starting_idx);
        const float& ty = perturbation.at<float>(starting_idx+1);
        const float& tz = perturbation.at<float>(starting_idx+2);
        const float& x_angle = perturbation.at<float>(starting_idx+3);
        const float cx = cos(x_angle);
        const float sx = sin(x_angle);
        const float& y_angle = perturbation.at<float>(starting_idx+4);
        const float cy = cos(y_angle);
        const float sy = sin(y_angle);
        const float& z_angle = perturbation.at<float>(starting_idx+5);
        const float cz = cos(z_angle);
        const float sz = sin(z_angle);
        const float T11 = T_matrix.at<float>(0,0);
        const float T12 = T_matrix.at<float>(0,1);
        const float T13 = T_matrix.at<float>(0,2);
        const float T14 = T_matrix.at<float>(0,3);
        const float T21 = T_matrix.at<float>(1,0);
        const float T22 = T_matrix.at<float>(1,1);
        const float T23 = T_matrix.at<float>(1,2);
        const float T24 = T_matrix.at<float>(1,3);
        const float T31 = T_matrix.at<float>(2,0);
        const float T32 = T_matrix.at<float>(2,1);
        const float T33 = T_matrix.at<float>(2,2);
        const float T34 = T_matrix.at<float>(2,3);
        const float T41 = T_matrix.at<float>(3,0);
        const float T42 = T_matrix.at<float>(3,1);
        const float T43 = T_matrix.at<float>(3,2);
        const float T44 = T_matrix.at<float>(3,3);

        // Apply the perturbation
        T_matrix.at<float>(0,0) = T31*sy + T41*tx + T11*cy*cz - T21*cy*sz;
        T_matrix.at<float>(0,1) = T32*sy + T42*tx + T12*cy*cz - T22*cy*sz;
        T_matrix.at<float>(0,2) = T33*sy + T43*tx + T13*cy*cz - T23*cy*sz;
        T_matrix.at<float>(0,3) = T34*sy + T44*tx + T14*cy*cz - T24*cy*sz;
        T_matrix.at<float>(1,0) = T41*ty + T11*(cx*sz + cz*sx*sy) + \
                                    T21*(cx*cz - sx*sy*sz) - T31*cy*sx;
        T_matrix.at<float>(1,1) = T42*ty + T12*(cx*sz + cz*sx*sy) + \
                                    T22*(cx*cz - sx*sy*sz) - T32*cy*sx;
        T_matrix.at<float>(1,2) = T43*ty + T13*(cx*sz + cz*sx*sy) + \
                                    T23*(cx*cz - sx*sy*sz) - T33*cy*sx;
        T_matrix.at<float>(1,3) = T44*ty + T14*(cx*sz + cz*sx*sy) + \
                                    T24*(cx*cz - sx*sy*sz) - T34*cy*sx;
        T_matrix.at<float>(2,0) = T41*tz + T11*(sx*sz - cx*cz*sy) + \
                                    T21*(cz*sx + cx*sy*sz) + T31*cx*cy;
        T_matrix.at<float>(2,1) = T42*tz + T12*(sx*sz - cx*cz*sy) + \
                                    T22*(cz*sx + cx*sy*sz) + T32*cx*cy;
        T_matrix.at<float>(2,2) = T43*tz + T13*(sx*sz - cx*cz*sy) + \
                                    T23*(cz*sx + cx*sy*sz) + T33*cx*cy;
        T_matrix.at<float>(2,3) = T44*tz + T14*(sx*sz - cx*cz*sy) + \
                                    T24*(cz*sx + cx*sy*sz) + T34*cx*cy;

    }


    // Invert a transformation matrix in a fast way by transposing the
    // rotational part and computing the translational part as
    // -R't
    cv::Mat invert_transformation_matrix(const cv::Mat& T_matrix) {

        cv::Mat T_matrix_inv = cv::Mat::eye(4,4,CV_32F);
        
        // Reference to rotational part (in transpose way)
        const float& R_11 = T_matrix.at<float>(0,0);
        const float& R_12 = T_matrix.at<float>(1,0);
        const float& R_13 = T_matrix.at<float>(2,0);
        const float& R_21 = T_matrix.at<float>(0,1);
        const float& R_22 = T_matrix.at<float>(1,1);
        const float& R_23 = T_matrix.at<float>(2,1);
        const float& R_31 = T_matrix.at<float>(0,2);
        const float& R_32 = T_matrix.at<float>(1,2);
        const float& R_33 = T_matrix.at<float>(2,2);

        // Reference to the translational part
        const float& t_x = T_matrix.at<float>(0,3);
        const float& t_y = T_matrix.at<float>(1,3);
        const float& t_z = T_matrix.at<float>(2,3);

        // Transpose the rotational part
        T_matrix_inv.at<float>(0,0) = R_11;
        T_matrix_inv.at<float>(0,1) = R_12;
        T_matrix_inv.at<float>(0,2) = R_13;
        T_matrix_inv.at<float>(1,0) = R_21;
        T_matrix_inv.at<float>(1,1) = R_22;
        T_matrix_inv.at<float>(1,2) = R_23;
        T_matrix_inv.at<float>(2,0) = R_31;
        T_matrix_inv.at<float>(2,1) = R_32;
        T_matrix_inv.at<float>(2,2) = R_33;

        // Compute the translational part
        T_matrix_inv.at<float>(0,3) = -(R_11*t_x + R_12*t_y + R_13*t_z);
        T_matrix_inv.at<float>(1,3) = -(R_21*t_x + R_22*t_y + R_23*t_z);
        T_matrix_inv.at<float>(2,3) = -(R_31*t_x + R_32*t_y + R_33*t_z);

        return T_matrix_inv;

    }



    /* 
    * This function takes a 3D point p and a costellation c of points
    * and return a pair:
    *   <idx of the nearest point to p in c, distance>
    */
    std::pair<int, float> nearest_3d_point(\
            const cv::Point3f& p, const std::vector<cv::Point3f>& c) {

        // Initialization
        const unsigned int& n_points = c.size();
        float current_distance;
        std::pair<int, float> result(-1, std::numeric_limits<float>::max());

        // For each point in the costellation c
        for(unsigned int i=0; i<n_points; ++i) {
            
            // Take the current point
            const cv::Point3f& p2 = c[i];

            // Compute distance
            current_distance = cv::norm(p-p2);

            // If it is the nearest one so far, save it
            if(current_distance < result.second) {
                result.first = i;
                result.second = current_distance;
            }
        }

        return result;

    }



    /*
    * This function, given a point (p_x, p_y) gives a vector of points
    * (nearest_points_ids) that filters out those points in the points vector
    * that are around the first given point (p_x, p_y). The threshold parameter
    * determine how near we want the points.
    */
    unsigned int nearest_2d_points(const float& p_x, const float& p_y, \
                            const std::vector<cv::KeyPoint>& points, \
                            std::vector<unsigned int>& nearest_points_ids, \
                            const float threshold) {
        
        // Initialization
        const unsigned int n_points = points.size();
        nearest_points_ids.clear();

        // Search for neighbors
        nearest_points_ids.reserve(n_points);
        for(unsigned int p_idx=0; p_idx<n_points; ++p_idx) {
            if(std::abs(p_x-points[p_idx].pt.x < threshold) && \
                std::abs(p_y-points[p_idx].pt.y < threshold))
                nearest_points_ids.emplace_back(p_idx);
        }
        nearest_points_ids.shrink_to_fit();

        return nearest_points_ids.size();

    }



    /*
    * Given two poses and a list of 3D points seen in common between them, this function 
    * computes the parallax between the two poses.
    * Inputs:
    *   pose1/pose2
    *   landmarks: the list of 3D points contained in the state
    *   common_landmarks_idx: the list of predicted 3D points ids that are seen in common between
    *       the two poses
    * Outputs:
    *   parallax
    */
    float computeParallax(const cv::Mat& pose1, const cv::Mat& pose2, \
                                const std::vector<Keypoint>& keypoints, \
                                const std::vector<unsigned int>& common_landmarks_ids) {
        
        // Initialization
        const unsigned int n_points = common_landmarks_ids.size();
        std::vector<float> parallaxesCos;
        cv::Mat normal1 = cv::Mat::zeros(3,1,CV_32F);
        cv::Mat normal2 = cv::Mat::zeros(3,1,CV_32F);
        float dist1, dist2;

        // Compute the origin of the pose 1
        const cv::Mat O1 = -pose1.rowRange(0,3).colRange(0,3).t() * \
                            pose1.rowRange(0,3).col(3);

        // Compute the origin of the pose2
        const cv::Mat O2 = -pose2.rowRange(0,3).colRange(0,3).t() * \
                            pose2.rowRange(0,3).col(3);

        // For each point
        parallaxesCos.reserve(n_points);
        for(unsigned int i=0; i<n_points; ++i) {

            // Take the current 3D point
            const cv::Point3f& current_point = keypoints[common_landmarks_ids[i]].getPosition();

            // Compute the normal origin-point for pose1
            normal1.at<float>(0,0) = current_point.x - O1.at<float>(0);
            normal1.at<float>(1,0) = current_point.y - O1.at<float>(1);
            normal1.at<float>(2,0) = current_point.z - O1.at<float>(2);

            // Compute the normal origin-point for pose2
            normal2.at<float>(0,0) = current_point.x - O2.at<float>(0);
            normal2.at<float>(1,0) = current_point.y - O2.at<float>(1);
            normal2.at<float>(2,0) = current_point.z - O2.at<float>(2);

            // Compute the distances pose-point
            dist1 = cv::norm(normal1);
            dist2 = cv::norm(normal2);

            // Compute the parallax cosine
            parallaxesCos.emplace_back( normal1.dot(normal2)/(dist1*dist2) );

        }
        parallaxesCos.shrink_to_fit();

        // Get the 50th smallest parallax
        std::sort(parallaxesCos.begin(), parallaxesCos.end());
        const float min_50th_parallax = \
            parallaxesCos[std::min(50, static_cast<int>(parallaxesCos.size() - 1))];
        
        // Compute the parallax between the two poses
        return std::acos(min_50th_parallax)*180 / CV_PI;

    }



    /*
    * Undistort the keypoints extracted from an image, using the
    * distorsion coefficients given from the camera calibration.
    */
    void undistort_keypoints(const std::vector<cv::KeyPoint>& keypoints, \
                            std::vector<cv::KeyPoint>& undistorted_keypoints, \
                            const cv::Mat& distorsion_coefficients, \
                            const cv::Mat& K) {
        
        // Initialization
        const unsigned int n_points = keypoints.size();
        undistorted_keypoints.reserve(n_points);
        cv::Mat keypoints_matrix(n_points,2,CV_32F);
        cv::Mat undistorted_keypoints_matrix(n_points,2,CV_32F);
        
        // Create a matrix of keypoints
        for(int i=0; i<n_points; i++) {
            keypoints_matrix.at<float>(i,0)=keypoints[i].pt.x;
            keypoints_matrix.at<float>(i,1)=keypoints[i].pt.y;
        }

        // Undistort points
        keypoints_matrix = keypoints_matrix.reshape(2);
        cv::undistortPoints(keypoints_matrix, keypoints_matrix,\
                            K, distorsion_coefficients, cv::Mat(), K);
        keypoints_matrix = keypoints_matrix.reshape(1);

        // Fill the undistorted keypoint vector
        for(int i=0; i<n_points; i++)
        {
            cv::KeyPoint undistorted_point = keypoints[i];
            undistorted_point.pt.x = keypoints_matrix.at<float>(i,0);
            undistorted_point.pt.y = keypoints_matrix.at<float>(i,1);
            undistorted_keypoints.emplace_back(undistorted_point);
        }
    }



    /*
    * This function, given a set of keypoints and a pose, returns
    * the median distance (along the z axis) of such points w.r.t. the given pose.
    */ 
    float compute_median_distance_cam_points(const std::vector<Keypoint>& points, \
                                            const cv::Mat& pose) {
        
        // Initialization
        const unsigned int n_points = points.size();
        std::vector<float> distances;
        const float& R31 = pose.at<float>(2,0);
        const float& R32 = pose.at<float>(2,1);
        const float& R33 = pose.at<float>(2,2);
        const float& tz = pose.at<float>(2,3);

        // Compute distance of each point
        distances.reserve(n_points);
        for(unsigned int i=0; i<n_points; ++i) {
            const cv::Point3f& current_point = points[i].getPosition();
            distances.emplace_back(\
                R31*current_point.x + R32*current_point.y + R33*current_point.z + tz \
            );
        }
        distances.shrink_to_fit();

        // Compute the median
        std::sort(distances.begin(), distances.end());
        return distances[(distances.size()-1)/2]/2;
    }



    /*
    * This function computes the distance between two poses represented
    * by two transformation matrices.
    */
    float compute_poses_distance(const cv::Mat& T1, const cv::Mat& T2) {

        return std::sqrt( \
            std::pow( T1.at<float>(0,3)-T2.at<float>(0,3), 2) + \
            std::pow( T1.at<float>(1,3)-T2.at<float>(1,3), 2) + \
            std::pow( T1.at<float>(2,3)-T2.at<float>(2,3), 2) \
        );
        
    }



    /*
    * This function computes the angle between two poses by using the
    * rotational part only and the functions derived from the Rodrigues'
    * formula.
    */  
    float compute_poses_angle(const cv::Mat& T1, const cv::Mat& T2) {

        // Initialization
        const cv::Mat R1 = T1.rowRange(0,3).colRange(0,3);
        const cv::Mat R2 = T2.rowRange(0,3).colRange(0,3);

        // Compute the distance between rotations
        const cv::Mat R = R1*R2.t();

        // Compute the angle
        return std::acos( (cv::trace(R)[0]-1.0)/2.0 );
    }



    /*
    * Computes the distnace between two 2D points.
    */
    float compute_distance_2d_points(const float& p1_x, const float& p1_y, \
                                    const float& p2_x, const float& p2_y) {
        return std::sqrt(std::pow(p2_x-p1_x, 2) + std::pow(p2_y-p1_y, 2));
    }
    
} // namespace SLucAM



// -----------------------------------------------------------------------------
// Representation conversion functions
// -----------------------------------------------------------------------------
namespace SLucAM {
    
    /*
    * This function, given a quaternion represented as a cv::Mat with
    * dim 4x1 and with this structure: [x; y; z; w], returns the
    * corresponding rotation matrix R.
    */
    void quaternion_to_matrix(const cv::Mat& quaternion, cv::Mat& R) {

        // Initialization
        R = cv::Mat::zeros(3,3,CV_32F);

        // Create the Eigen quaternion
        Eigen::Quaternionf Eigen_quaternion;
        Eigen_quaternion.x() = quaternion.at<float>(0,0);
        Eigen_quaternion.y() = quaternion.at<float>(1,0);
        Eigen_quaternion.z() = quaternion.at<float>(2,0);
        Eigen_quaternion.w() = quaternion.at<float>(3,0);

        // Convert it to R
        Eigen::Matrix3f Eigen_R = Eigen_quaternion.normalized().toRotationMatrix();

        // Back to cv representation
        R.at<float>(0,0) = Eigen_R(0,0);
        R.at<float>(0,1) = Eigen_R(0,1);
        R.at<float>(0,2) = Eigen_R(0,2);
        R.at<float>(1,0) = Eigen_R(1,0);
        R.at<float>(1,1) = Eigen_R(1,1);
        R.at<float>(1,2) = Eigen_R(1,2);
        R.at<float>(2,0) = Eigen_R(2,0);
        R.at<float>(2,1) = Eigen_R(2,1);
        R.at<float>(2,2) = Eigen_R(2,2);
    }



    /*
    * This function, given a rotation matrix represented as a cv::Mat, 
    * returns the corresponding quaternion represented as a cv::Mat with 
    * dim 4x1 and with this structure: [x; y; z; w].
    */
    void matrix_to_quaternion(const cv::Mat& R, cv::Mat& quaternion) {

        // Initialization
        quaternion = cv::Mat::zeros(4,1,CV_32F);

        // Create the Eigen matrix R
        Eigen::Matrix3f Eigen_R(3,3);
        Eigen_R(0,0) = R.at<float>(0,0);
        Eigen_R(0,1) = R.at<float>(0,1);
        Eigen_R(0,2) = R.at<float>(0,2);
        Eigen_R(1,0) = R.at<float>(1,0);
        Eigen_R(1,1) = R.at<float>(1,1);
        Eigen_R(1,2) = R.at<float>(1,2);
        Eigen_R(2,0) = R.at<float>(2,0);
        Eigen_R(2,1) = R.at<float>(2,1);
        Eigen_R(2,2) = R.at<float>(2,2);
        
        // Convert it to quaternion
        Eigen::Quaternionf Eigen_quaternion(Eigen_R);

        // Back to cv representation
        quaternion.at<float>(0,0) = Eigen_quaternion.x();
        quaternion.at<float>(1,0) = Eigen_quaternion.y();
        quaternion.at<float>(2,0) = Eigen_quaternion.z();
        quaternion.at<float>(3,0) = Eigen_quaternion.w();
    }



    /*
    * This function, given a transformation matrix, returns the same matrix
    * represented as a SE3Quat for g2o.
    */
    g2o::SE3Quat transformation_matrix_to_SE3Quat(const cv::Mat& T_matrix) {
        Eigen::Matrix<double,3,3> R;
        R << T_matrix.at<float>(0,0), T_matrix.at<float>(0,1), T_matrix.at<float>(0,2),
            T_matrix.at<float>(1,0), T_matrix.at<float>(1,1), T_matrix.at<float>(1,2),
            T_matrix.at<float>(2,0), T_matrix.at<float>(2,1), T_matrix.at<float>(2,2);

        Eigen::Matrix<double,3,1> t(T_matrix.at<float>(0,3), \
                                    T_matrix.at<float>(1,3), \
                                    T_matrix.at<float>(2,3));

        return g2o::SE3Quat(R,t);
    }



    /*
    * This function, given a SE3Quat pose, returns the same matrix represented
    * as a OpenCV transformation matrix.
    */
    cv::Mat SE3Quat_to_transformation_matrix(const g2o::SE3Quat& se3quat) {
        cv::Mat T_matrix = cv::Mat::zeros(4,4,CV_32F);
        Eigen::Matrix<double,4,4> eigen_T_matrix = se3quat.to_homogeneous_matrix();
        for(int i=0;i<4;i++)
            for(int j=0; j<4; j++)
                T_matrix.at<float>(i,j) = eigen_T_matrix(i,j);
        return T_matrix;
    }



    /*
    * This function, given a Point3f (float) in OpenCV representation, returns the same
    * point in Eigen Matrix (double) representation: useful for g2o.
    */
    Eigen::Matrix<double,3,1> point_3d_to_vector_3d(const cv::Point3f& point) {
        Eigen::Matrix<double,3,1> v;
        v << point.x, point.y, point.z;
        return v;
    }



    /*
    * This function, given a Eigen 3x1 vector (double), returns the same vector
    * as a 3d point in OpenCV representation (Point3f).
    */
    cv::Point3f vector_3d_to_point_3d(const Eigen::Matrix<double,3,1>& vector) {
        return cv::Point3f(vector(0), vector(1), vector(2));
    }




    /*
    * This function, given a Keypoint (float) in OpenCV representation, returns the same
    * point in Eigen Matrix (double) representation: useful for g2o.
    */
    Eigen::Matrix<double,2,1> point_2d_to_vector_2d(const cv::KeyPoint& point) {
        Eigen::Matrix<double,2,1> v;
        v << point.pt.x, point.pt.y;
        return v;
    }

} // namespace SLucAM



// -----------------------------------------------------------------------------
// Implementation of multi-view geometry functions
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * Function that, given a set of, at least, 8 couples of points, estimate
    * the Foundamental matrix (implementation of the 8-point algorithm).
    * Better if the points p_img1 and p_img2 are normalized, in such case
    * please de-normalize F after this function.
    * Inputs:
    *   p_img1/p_img2: input points
    *   matches: the correspondances between the two set of points
    *   idxs: we will compute the F matrix only on those points contained in 
    *           the matches vector, for wich we have indices in this vector idxs
    *   F: estimated Foundamental Matrix
    */
    void estimate_foundamental(const std::vector<cv::KeyPoint>& p_img1, \
                                const std::vector<cv::KeyPoint>& p_img2, \
                                const std::vector<cv::DMatch>& matches, \
                                const std::vector<unsigned int>& idxs, \
                                cv::Mat& F) {
        
        // Initialization
        unsigned int n_points = idxs.size();

        // Ensemble the A matrix for equations
        Eigen::MatrixXf A(n_points, 9);
        for(unsigned int i = 0; i<n_points; ++i) {
            const float& p1_x = p_img1[matches[idxs[i]].queryIdx].pt.x;
            const float& p1_y = p_img1[matches[idxs[i]].queryIdx].pt.y;
            const float& p2_x = p_img2[matches[idxs[i]].trainIdx].pt.x;
            const float& p2_y = p_img2[matches[idxs[i]].trainIdx].pt.y;
            A(i,0) = p1_x * p2_x;
            A(i,1) = p1_y * p2_x;
            A(i,2) = p2_x;
            A(i,3) = p1_x * p2_y;
            A(i,4) = p1_y * p2_y;
            A(i,5) = p2_y;
            A(i,6) = p1_x;
            A(i,7) = p1_y;
            A(i,8) = 1;
        }

        // Compute the linear least square solution
        Eigen::JacobiSVD<Eigen::MatrixXf> svd_A(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix<float,3,3,Eigen::RowMajor> unconstrained_F(svd_A.matrixV().col(8).data());

        // Constrain F making the rank 2 by zeroing the last
        // singular value
        Eigen::JacobiSVD<Eigen::Matrix3f> svd_F(unconstrained_F, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector3f w = svd_F.singularValues();
        w(2) = 0;
        Eigen::Matrix3f F_eigen = svd_F.matrixU() * Eigen::DiagonalMatrix<float,3>(w) * svd_F.matrixV().transpose();

        // Compute the OpenCV version of F
        cv::eigen2cv(F_eigen, F);
    }



    /*
    * Function that, given a set of, at least, 4 couples of points, estimate
    * the Homography (implementation of the DLT algorithm).
    * We assume that the points p_img1 and p_img2 are normalized and,
    * please, de-normalize H after this function.
    * Inputs:
    *   p_img1/p_img2: input points
    *   H: estimated Homography
    *   matches: the correspondances between the two set of points
    *   idxs: we will compute the H matrix only on those points contained in 
    *           the matches vector, for wich we have indices in this vector idxs
    *           In particular we will consider only the first 4 elements of it
    *           (the minimal set to compute H)
    */
    void estimate_homography(const std::vector<cv::KeyPoint>& p_img1, \
                                const std::vector<cv::KeyPoint>& p_img2, \
                                const std::vector<cv::DMatch>& matches, \
                                const std::vector<unsigned int>& idxs, \
                                cv::Mat& H) {
        
        // Initialization
        unsigned int n_points = 4; // We use only the first 4 points in idxs

        // Ensemble the A matrix
        cv::Mat A = cv::Mat::zeros(2*n_points,9,CV_32F);
        for(int i=0; i<n_points; i++)
        {
            const float& p1_x = p_img1[matches[idxs[i]].queryIdx].pt.x;
            const float& p1_y = p_img1[matches[idxs[i]].queryIdx].pt.y;
            const float& p2_x = p_img2[matches[idxs[i]].trainIdx].pt.x;
            const float& p2_y = p_img2[matches[idxs[i]].trainIdx].pt.y;

            A.at<float>(2*i,3) = -p1_x;
            A.at<float>(2*i,4) = -p1_y;
            A.at<float>(2*i,5) = -1;
            A.at<float>(2*i,6) = p2_y*p1_x;
            A.at<float>(2*i,7) = p2_y*p1_y;
            A.at<float>(2*i,8) = p2_y;
            A.at<float>(2*i+1,0) = p1_x;
            A.at<float>(2*i+1,1) = p1_y;
            A.at<float>(2*i+1,2) = 1;
            A.at<float>(2*i+1,6) = -p2_x*p1_x;
            A.at<float>(2*i+1,7) = -p2_x*p1_y;
            A.at<float>(2*i+1,8) = -p2_x;
        }

        // compute SVD of A
        cv::Mat u,w,vt;
        cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

        // Use the last column of V to build the h vector and, reshaping it
        // we obtain the homography matrix
        H = vt.row(8).reshape(0, 3);
    }
    
} // namespace SLucAM



// -----------------------------------------------------------------------------
// Projective ICP functions implementation
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * Function that computes the error and jacobian for Projective ICP
    * Inputs:
    *   guessed_pose: guessed pose of the world w.r.t. camera
    *   guessed_landmark: point in the space where we think is located the 
    *           measured landmark
    *   measured_point: measured point on the projective plane
    *   K: camera matrix
    *   img_rows: #rows in the image plane pixels matrix
    *   img_cols: #cols in the image plane pixels matrix
    *   error: (output) we assume it is already initialized as a 2x1 
    *       matrix
    *   J: jacobian matrix of the error w.r.t. guessed_pose (output), 
    *       we assume it is already initialized as a 2x6 matrix
    * Outputs:
    *   true if the projection is valid, false otherwise
    */
    bool error_and_jacobian_Posit(const cv::Mat& guessed_pose, \
                                const cv::Point3f& guessed_landmark, \
                                const cv::KeyPoint& measured_point, \
                                const cv::Mat& K, \
                                const float& img_rows, \
                                const float& img_cols, \
                                cv::Mat& error, cv::Mat& J) {
        
        // Some reference to save time+
        const float& K_11 = K.at<float>(0,0);
        const float& K_12 = K.at<float>(0,1);
        const float& K_13 = K.at<float>(0,2);
        const float& K_21 = K.at<float>(1,0);
        const float& K_22 = K.at<float>(1,1);
        const float& K_23 = K.at<float>(1,2);
        const float& K_31 = K.at<float>(2,0);
        const float& K_32 = K.at<float>(2,1);
        const float& K_33 = K.at<float>(2,2);
        const float& X_11 = guessed_pose.at<float>(0,0);
        const float& X_12 = guessed_pose.at<float>(0,1);
        const float& X_13 = guessed_pose.at<float>(0,2);
        const float& X_14 = guessed_pose.at<float>(0,3);
        const float& X_21 = guessed_pose.at<float>(1,0);
        const float& X_22 = guessed_pose.at<float>(1,1);
        const float& X_23 = guessed_pose.at<float>(1,2);
        const float& X_24 = guessed_pose.at<float>(1,3);
        const float& X_31 = guessed_pose.at<float>(2,0);
        const float& X_32 = guessed_pose.at<float>(2,1);
        const float& X_33 = guessed_pose.at<float>(2,2);
        const float& X_34 = guessed_pose.at<float>(2,3);
        const float& X_41 = guessed_pose.at<float>(3,0);
        const float& X_42 = guessed_pose.at<float>(3,1);
        const float& X_43 = guessed_pose.at<float>(3,2);
        const float& X_44 = guessed_pose.at<float>(3,3);
        const float& P_x = guessed_landmark.x;
        const float& P_y = guessed_landmark.y;
        const float& P_z = guessed_landmark.z;

        // Compute the position of the point w.r.t. camera frame
        const float p_cam_x = (X_11*P_x + X_12*P_y + X_13*P_z) + X_14;
        const float p_cam_y = (X_21*P_x + X_22*P_y + X_23*P_z) + X_24;
        const float p_cam_z = (X_31*P_x + X_32*P_y + X_33*P_z) + X_34; 

        // Check if the prediction is in front of the camera
        if(p_cam_z < 0) return false;

        // Compute the prediction (projection)
        const float p_camK_x = K_11*p_cam_x + K_12*p_cam_y + K_13*p_cam_z;
        const float p_camK_y = K_21*p_cam_x + K_22*p_cam_y + K_23*p_cam_z;
        const float p_camK_z = K_31*p_cam_x + K_32*p_cam_y + K_33*p_cam_z;
        const float iz = 1.0/(p_camK_z);
        const float z_hat_x = p_camK_x*iz;
        const float z_hat_y = p_camK_y*iz;

        // Check if the point prediction on projection plane is inside 
        // the camera frustum
        // TODO: assicurati che img_cols e img_rows siano corretti
        if (z_hat_x < 0 || 
            z_hat_x > img_cols-1 ||
            z_hat_y < 0 || 
            z_hat_y > img_rows-1)
            return false;
                
        // Compute the error
        error.at<float>(0) = z_hat_x - measured_point.pt.x;
        error.at<float>(1) = z_hat_y - measured_point.pt.y;

        // Compute the Jacobian
        const float iz2 = iz*iz;
        const float p_cam_iz2_x = -p_camK_x*iz2;
        const float p_cam_iz2_y = -p_camK_y*iz2;

        J.at<float>(0,0) = K_11*iz + K_31*p_cam_iz2_x;
        J.at<float>(0,1) = K_12*iz + K_32*p_cam_iz2_x;
        J.at<float>(0,2) = K_13*iz + K_33*p_cam_iz2_x;
        J.at<float>(0,3) = p_cam_y*(K_13*iz + K_33*p_cam_iz2_x) - \
                            p_cam_z*(K_12*iz + K_32*p_cam_iz2_x);
        J.at<float>(0,4) = p_cam_z*(K_11*iz + K_31*p_cam_iz2_x) - \
                            p_cam_x*(K_13*iz + K_33*p_cam_iz2_x);
        J.at<float>(0,5) = p_cam_x*(K_12*iz + K_32*p_cam_iz2_x) - \
                            p_cam_y*(K_11*iz + K_31*p_cam_iz2_x);
        J.at<float>(1,0) = K_21*iz + K_31*p_cam_iz2_y;
        J.at<float>(1,1) = K_22*iz + K_32*p_cam_iz2_y;
        J.at<float>(1,2) = K_23*iz + K_33*p_cam_iz2_y;
        J.at<float>(1,3) = p_cam_y*(K_23*iz + K_33*p_cam_iz2_y) - \
                            p_cam_z*(K_22*iz + K_32*p_cam_iz2_y);
        J.at<float>(1,4) = p_cam_z*(K_21*iz + K_31*p_cam_iz2_y) - \
                            p_cam_x*(K_23*iz + K_33*p_cam_iz2_y);
        J.at<float>(1,5) = p_cam_x*(K_22*iz + K_32*p_cam_iz2_y) - \
                            p_cam_y*(K_21*iz + K_31*p_cam_iz2_y);

        return true;

    }


    /*
    * Function that perform, given a measurement taken from a camera,
    * the projective ICP to get the pose of the world w.r.t. the camera
    * from which such measurement are taken. There will be taken in consideration
    * only such measurements for which the pose of the landmark is already
    * triangulated (so guessed)
    * Inputs:
    *   guessed_pose: initial guess (and output)
    *   measurement: the measurement for which we need to predict the pose
    *   points_associations_filter: it will contain in position i true if
    *           the element in position i of the points_associations vector
    *           is an inlier, false otherwise
    *   points_associations: list of associations 2D point <-> 3D point
    *   landmarks: set of triangulated landmarks
    *   K: camera matrix
    *   n_iterations: #iterations to perform for Posit
    *   kernel_threshold: threshold for the outliers
    *   threshold_to_ignore: error threshold that determine if an outlier 
    *           is too outlier to be considered
    * Outputs:
    *   n_inliers of the last iteration
    */
    unsigned int perform_Posit(cv::Mat& guessed_pose, \
                                const Measurement& measurement, \
                                std::vector<bool>& points_associations_filter, \
                                const std::vector<std::pair<unsigned int, \
                                        unsigned int>>& points_associations, \
                                const std::vector<Keypoint>& keypoints, \
                                const cv::Mat& K, \
                                const float& kernel_threshold, \
                                const float& threshold_to_ignore, \
                                const unsigned int n_iterations, \
                                const float damping_factor) {
        
        // Initialization
        const unsigned int n_observations = points_associations.size();
        const float img_rows = 2*K.at<float>(1, 2);
        const float img_cols = 2*K.at<float>(0, 2);
        float current_chi = 0.0;
        std::vector<unsigned int> n_inliers(n_iterations, 0);
        std::vector<float> chi_stats(n_iterations, 0.0);
        cv::Mat H, b;
        cv::Mat error = cv::Mat::zeros(2,1,CV_32F);
        cv::Mat J = cv::Mat::zeros(2,6,CV_32F);
        const cv::Mat DampingMatrix = \
                    cv::Mat::eye(6, 6, CV_32F)*damping_factor;

        // For each iteration
        for(unsigned int iter=0; iter<n_iterations; ++iter) {
            
            // Reset H and b
            H = cv::Mat::zeros(6,6,CV_32F);
            b = cv::Mat::zeros(6,1,CV_32F);

            // For each observation
            for(unsigned int obs_idx=0; obs_idx<n_observations; ++obs_idx) {

                // Take the measured 2D point for the current observation
                const cv::KeyPoint& measured_point = \
                        measurement.getPoints()[points_associations[obs_idx].first];

                // Take the guessed landmark position of the current observation
                const cv::Point3f& guessed_landmark = \
                        keypoints[points_associations[obs_idx].second].getPosition();
                
                // Compute error and jacobian
                if(!error_and_jacobian_Posit(guessed_pose, guessed_landmark, \
                                                measured_point, K, img_rows, \
                                                img_cols, error, J)) {
                    points_associations_filter[obs_idx] = false;
                    continue;   // Discard not valid projections
                }
                    

                // Compute chi error
                const float& e_1 = error.at<float>(0);
                const float& e_2 = error.at<float>(1);
                current_chi = (e_1*e_1) + (e_2*e_2);

                // Deal with outliers
                if(current_chi > threshold_to_ignore){
                    points_associations_filter[obs_idx] = false;
                    continue;
                }

                // Robust kernel
                if(current_chi > kernel_threshold) {
                    error *= sqrt(kernel_threshold/current_chi);
                    current_chi = kernel_threshold;
                    points_associations_filter[obs_idx] = false;
                } else {
                    ++n_inliers[iter];
                    points_associations_filter[obs_idx] = true;
                }

                // Update chi stats
                chi_stats[iter] += current_chi;

                // Update H and b
                H += J.t()*J;
                b += J.t()*error;

            }

            // Damping the H matrix
            H += DampingMatrix;

            // Solve linear system to get the perturbation
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> \
                        H_Eigen(H.ptr<float>(), H.rows, H.cols);
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> \
                        b_Eigen(b.ptr<float>(), b.rows, b.cols);
            Eigen::VectorXf dx_Eigen = H_Eigen.ldlt().solve(-b_Eigen);
            cv::Mat dx(dx_Eigen.rows(), dx_Eigen.cols(), CV_32F, dx_Eigen.data());

            // Apply the perturbation
            apply_perturbation_Tmatrix(dx, guessed_pose, 0);
        }

        return n_inliers.back();

    }

} // namespace SLucAM