//
// SLucAM_state.h
//
// In this module we have all the function to deal with the state and the
// state class itself.
//


#ifndef SLUCAM_STATE_H
#define SLUCAM_STATE_H

// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <opencv2/features2d.hpp>
#include <SLucAM_keyframe.h>
#include <SLucAM_measurement.h>
#include <SLucAM_matcher.h>
#include <SLucAM_keypoint.h>
#include <map>
#include <iostream>



// -----------------------------------------------------------------------------
// State class
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * This class represent the state of the system. It mantains all the poses
    * accumulated so far and all the triangulated "landmarks" positions.
    */
    class State {
    
    public:

        State(const unsigned int keyframe_density);

        State(cv::Mat& K, std::vector<Measurement>& measurements, \
            const unsigned int expected_poses, \
            const unsigned int expected_landmarks, \
            const unsigned int keyframe_density);
        
        State(cv::Mat& K, cv::Mat& distorsion_coefficients, \
            std::vector<Measurement>& measurements, \
            const unsigned int expected_poses, \
            const unsigned int expected_landmarks, \
            const unsigned int keyframe_density);
        
        bool initializeState(Matcher& matcher, \
                            const bool verbose=false);
        
        bool integrateNewMeasurement(Matcher& matcher, \
                                        const bool& triangulate_new_points, \
                                        const float& kernel_threshold_POSIT, \
                                        const float& inliers_threshold_POSIT, \
                                        const bool verbose);
        
        void performTotalBA(const unsigned int& n_iters, const bool verbose=false);

        void performLocalBA(const unsigned int& n_iters, const bool verbose=false);

        const unsigned int reaminingMeasurements() const {
            return (this->_measurements.size() - this->_next_measurement_idx);
        };

        void addKeyFrame(const unsigned int& meas_idx, const unsigned int& pose_idx, \
                        const int& observer_keyframe_idx, const bool verbose=false);

        bool canBeSpawnedAsKeyframe(const cv::Mat& pose, \
                                    const std::vector<std::pair<unsigned int, unsigned int>> \
                                        points_associations, \
                                    const bool verbose=false);

        void getLocalMap(const unsigned int& keyframe_idx, \
                            std::vector<unsigned int>& observed_keypoints, \
                            std::vector<unsigned int>& near_local_keyframes, \
                            std::vector<unsigned int>& far_local_keyframes);
        
        Measurement& getNextMeasurement() {
            this->_from_last_keyframe++;
            return this->_measurements[this->_next_measurement_idx++];
        };

        Measurement& getFirstMeasurement() {
            Measurement& first_meas = this->_measurements[this->_first_meas_for_initialization++];
            this->_next_measurement_idx = this->_first_meas_for_initialization;
            return first_meas;
        }
        
        const Measurement& getLastMeasurement() const {
            if(this->_next_measurement_idx > 0)
                return this->_measurements[this->_next_measurement_idx-1];
            else
                return this->_measurements[0];
        };
        
        const cv::Mat& getCameraMatrix() const \
            {return this->_K;};
        
        const cv::Mat& getDistorsionCoefficients() const \
            {return this->_distorsion_coefficients;};

        const std::vector<Measurement>& getMeasurements() const \
            {return this->_measurements;};

        const std::vector<cv::Mat>& getPoses() const \
            {return this->_poses;};

        const std::vector<Keypoint>& getKeypoints() const \
            {return this->_keypoints;};

        const std::vector<Keyframe>& getKeyframes() const \
            {return this->_keyframes;};
    
    private: 
        
        static bool predictPose(cv::Mat& guessed_pose, \
                                const Measurement& meas_to_predict, \
                                const unsigned int& last_meas_idx, \
                                std::vector<std::pair<unsigned int, unsigned int>>& \
                                        points_associations, \
                                Matcher& matcher, \
                                const std::vector<Keyframe>& keyframes, \
                                const std::vector<Keypoint>& keypoints, \
                                const std::vector<Measurement>& measurements, \
                                const std::vector<cv::Mat>& poses, \
                                const cv::Mat& K, \
                                const float& kernel_threshold_POSIT, \
                                const float& inliers_threshold_POSIT, \
                                const bool verbose=false);

        static void triangulateNewPoints(std::vector<Keyframe>& keyframes, \
                                        std::vector<Keypoint>& keypoints, \
                                        std::vector<Measurement>& measurements, \
                                        const std::vector<cv::Mat>& poses, \
                                        Matcher& matcher, \
                                        const cv::Mat& K, \
                                        const bool verbose=false);

        static void associateNewKeypoints(const std::vector<cv::Point3f>& predicted_landmarks, \
                                        const std::vector<cv::DMatch>& matches, \
                                        const std::vector<unsigned int>& matches_filter, \
                                        std::vector<Keypoint>& keypoints, \
                                        std::vector<std::pair<unsigned int, \
                                                unsigned int>>& meas1_points_associations, \
                                        std::vector<std::pair<unsigned int, \
                                                unsigned int>>& meas2_points_associations, \
                                        const bool verbose=false);
        
        static void addPointsAssociations(const unsigned int& meas_idx, \
                                            const std::vector<std::pair<unsigned int, unsigned int>>& \
                                                points_associations, \
                                            std::vector<Measurement>& measurements, \
                                            std::vector<Keypoint>& keypoints);
        
        static bool containsLandmark(const std::vector<std::pair<unsigned int, \
                                        unsigned int>>& points_associations, \
                                        const unsigned int& landmark_idx);
                                        
        static void findInitialAssociations(const Measurement& meas, \
                                            const unsigned int& last_meas_idx, \
                                            std::vector<std::pair<unsigned int, unsigned int>>& points_associations, \
                                            Matcher& matcher, \
                                            const std::vector<Measurement>& measurements, \
                                            const std::vector<Keypoint>& keypoints, \
                                            const unsigned int& window_size=3);

        static void projectFromMeasurement(const Measurement& meas, \
                                            const cv::Mat& T, const cv::Mat& K, \
                                            const std::vector<Keypoint>& keypoints, \
                                            const std::vector<unsigned int>& local_keypoints_ids, \
                                            std::vector<std::pair<unsigned int, unsigned int>>& \
                                                    points_associations);

        // Camera matrix and distorsion coefficients
        cv::Mat _K;
        cv::Mat _distorsion_coefficients;

        // The vector containing all the measurements, ordered by time
        std::vector<Measurement> _measurements;
     
        // The vector containing all the poses, ordered by time (poses of
        // the world wrt cameras)
        std::vector<cv::Mat> _poses;

        // The vector containing all the triangulated points, ordered
        // by insertion
        std::vector<Keypoint> _keypoints;

        // This vector contains all the keyframe
        std::vector<Keyframe> _keyframes;

        // Reference to the next measurement to analyze
        unsigned int _next_measurement_idx;

        // The reference to the first measurement for initialization
        unsigned int _first_meas_for_initialization;

        // Count how many measurements we integrate from the last keyframe
        unsigned int _from_last_keyframe;

        // How many frames we should wait before to spawn a new keyframe
        unsigned int _keyframe_density;
    };

} // namespace SLucAM



#endif // SLUCAM_STATE_H