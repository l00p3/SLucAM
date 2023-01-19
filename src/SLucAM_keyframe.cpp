//
// SLucAM_keyframe.h implementation
//


// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <SLucAM_keyframe.h>



// -----------------------------------------------------------------------------
// Implementation of Keyframe class methods
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * Basic constructor
    */
    Keyframe::Keyframe(const unsigned int& meas_idx, \
                        const unsigned int& pose_idx, \
                        const std::vector<unsigned int>& local_keypoints, \
                        const std::vector<unsigned int>& local_keyframes) {
        this->_meas_idx = meas_idx;
        this->_pose_idx = pose_idx;
        this->_local_keypoints = local_keypoints;
        this->_local_keyframes = local_keyframes;
    }

} // namespace SLucAM