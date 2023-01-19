//
// SLucAM_keypoint.h implementation
//


// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <SLucAM_keypoint.h>
#include <SLucAM_matcher.h>
#include <algorithm>
#include <iostream>
#include <limits>



// -----------------------------------------------------------------------------
// Implementation of Keypoint class methods
// -----------------------------------------------------------------------------
namespace SLucAM {


    /*
    * This function find, among all the descriptors of the points that
    * observe the current keypoint, the descriptor that has the minor
    * median distance to the rest of keypoints.
    */
    void Keypoint::updateDescriptor(const std::vector<Measurement>& measurements) {

        // Initialization
        const unsigned int n_observers = this->_observers.size();
        std::vector<std::vector<float>> distances(n_observers, \
                                            std::vector<float>(n_observers, 0));
        float best_distance = std::numeric_limits<float>::max();
        float current_distance;
        unsigned int best_idx = 0;

        // Compute mutual distances
        for(unsigned int i=0; i<n_observers; ++i) {
            for(unsigned int j=0; j<n_observers; ++j) {
                distances[i][j] = Matcher::compute_descriptors_distance(\
                        this->getObserverDescriptor(measurements, i), 
                        this->getObserverDescriptor(measurements, j));
            }
        }
        
        // Save as representative descriptor the descriptor with the
        // lower median distance
        for(unsigned int i=0; i<n_observers; ++i){
            std::vector<float>& row = distances[i];
            std::sort(row.begin(), row.end());
            current_distance = row[n_observers/2];
            if(current_distance < best_distance) {
                best_distance = current_distance;
                best_idx = i;
            }
        }
        this->_descriptor = this->getObserverDescriptor(measurements, best_idx);

    }



    /*
    * This function, given an idx of an observer, returns the associated
    * descriptor.
    */
    const cv::Mat Keypoint::getObserverDescriptor(const std::vector<Measurement>& measurements, \
                                                const unsigned int& idx) {
        return measurements[this->_observers[idx].first].getDescriptor(this->_observers[idx].second);
    }
    
} // namespace SLucAM