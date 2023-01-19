//
// SLucAM_keyframe.h
//
// In this module we have all the function to deal with the concept of keyframe.
//


#ifndef SLUCAM_KEYFRAME_H
#define SLUCAM_KEYFRAME_H

// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <vector>
#include <iostream>
#include <set>



// -----------------------------------------------------------------------------
// Keyframe class
// -----------------------------------------------------------------------------
namespace SLucAM {
    
    /*
    * This class take note of each KeyFrame in the state. Basically it contains
    * informations about:
    *   - which measurement in the state is the measurement from which
    *       this keyframe is taken
    *   - which pose in the state is the predicted pose for this keyframe
    *   - a vector of Map <point_idx, landmark_idx> that associates at each point
    *       in the measurement to which this keyframe refers, the 3D 
    *       predicted landmark in the state
    *   - a vector that contains all the keyframe(poses) that the current Keyframe 
    *       observes
    *   - a vector that contains all the keyframes(poses) that observes the current
    *       keyframe
    */
    class Keyframe {

    public:

        Keyframe(const unsigned int& meas_idx, \
                const unsigned int& pose_idx, \
                const std::vector<unsigned int>& local_keypoints, \
                const std::vector<unsigned int>& local_keyframes);

        void addKeyframeObserved(const int& pose_idx) {
            this->_keyframes_observed.emplace_back(pose_idx);
        }

        void addObserverKeyframe(const int& pose_idx) {
            this->_observers_keyframes.emplace_back(pose_idx);
        }

        const unsigned int& getPoseIdx() const {return this->_pose_idx;}

        const unsigned int& getMeasIdx() const {return this->_meas_idx;}

        const std::vector<unsigned int>& getKeyframesObserved() const {
            return this->_keyframes_observed;
        }

        const std::vector<unsigned int>& getObserversKeyframes() const {
            return this->_observers_keyframes;
        }

        const std::vector<unsigned int>& getLocalKeypoints() const {
            return this->_local_keypoints;
        }

        const std::vector<unsigned int>& getLocalKeyframes() const {
            return this->_local_keyframes;
        }

    private:

        // Idx of the measure of the keyframe (referred to the list of measurements
        // in the state)
        unsigned int _meas_idx;

        // Idx of the pose of the keyframe (referred to the list of poses
        // in the state)
        unsigned int _pose_idx;
        
        // List of keyframes that this observes
        std::vector<unsigned int> _keyframes_observed;

        // List of keyframes that observes this
        std::vector<unsigned int> _observers_keyframes;

        // List of local keypoints
        std::vector<unsigned int> _local_keypoints;

        // List of local keyframes
        std::vector<unsigned int> _local_keyframes;

    };

} // namespace SLucAM



#endif // SLUCAM_KEYFRAME_H