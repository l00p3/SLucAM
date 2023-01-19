//
// SLucAM_measurement.h implementation
//


// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <SLucAM_measurement.h>
#include <SLucAM_image.h>
#include <SLucAM_geometry.h>



// -----------------------------------------------------------------------------
// Implementation of the measurement class methods
// -----------------------------------------------------------------------------
namespace SLucAM {

    // Initialize the static member
    unsigned int Measurement::_next_id = 0;

    /*
    * Constructor of the class Measurement. It takes the vector of points
    * for a given measurements and the relative image descriptors. It
    * normalizes the points also
    */
    Measurement::Measurement(std::vector<cv::KeyPoint>& points, \
                            cv::Mat& descriptors) {
        
        // Store the points
        this->_points = points;
        this->_descriptors = descriptors;

        // Assign the id to the measurement
        this->_meas_id = Measurement::_next_id;
        Measurement::_next_id++;

        // Initialize the points associations
        this->_points2keypoints = \
            std::vector<unsigned int>(this->_points.size(), -1);
        
        // Initialize the keyframe id to NOT A KEYFRAME
        this->_keyframe_idx = -1;
    }



    /*
    * Gives a descriptor of a point, determined by its idx.
    */
    cv::Mat Measurement::getDescriptor(const unsigned int& idx) const {
        return this->_descriptors.row(idx);
    }



    /*
    * Simply add a set of points associations.
    */
    void Measurement::addAssociations(const std::vector<std::pair<unsigned int, unsigned int>>& \
                                        new_points_associations) {
        for(const auto& el: new_points_associations)
            this->_points2keypoints[el.first] = el.second;
    }



    /*
    * Returns all the points associations as a vector of pairs 2D,3D points
    */
    void Measurement::getPointsAssociations(std::vector<std::pair<unsigned int, unsigned int>>& \
                                        points_associations) const {
        const unsigned int n_points = this->_points2keypoints.size();
        points_associations.clear(); points_associations.reserve(n_points);
        for(unsigned int i=0; i<n_points; ++i) {
            const unsigned int& p_3d_idx = this->_points2keypoints[i];
            if(p_3d_idx != -1)
                points_associations.emplace_back(i, p_3d_idx);
        }
        points_associations.shrink_to_fit();
    }



    /*
    * Computes the list of ids of keypoints seen from both the current
    * measurement and the meas2.
    */
    void Measurement::getCommonKeypoints(const Measurement& meas2, \
                                            std::vector<unsigned int>& commond_keypoints_ids) const {
        
        // Create a "counter" for keypoints
        std::map<unsigned int, unsigned int> counter;
        for(const auto& el: this->_points2keypoints)
            if(el != -1)
                counter[el]++;
        for(const auto& el: meas2._points2keypoints)
            if(el != -1)
                counter[el]++;
        
        // Save only keypoints seen from both
        commond_keypoints_ids.clear();
        commond_keypoints_ids.reserve(counter.size());
        for(const auto& el: counter)
            if(el.second > 1)
                commond_keypoints_ids.emplace_back(el.first);
        commond_keypoints_ids.shrink_to_fit();

    }



    /*
    * This function returns the list of 3D point observed by the current measurement
    */
    void Measurement::getObservedPoints(std::vector<unsigned int>& ids) const {
       ids.clear(); ids.reserve(this->_points.size());
       for(const auto& el: this->_points2keypoints) {
           if(el != -1) ids.emplace_back(el);
       }
       ids.shrink_to_fit();
   }



   /*
    * Adds to the seen_keypoints_set set all the keypoints seen from the current
    * measurement.
    */
    void Measurement::getObservedPointsSet(std::set<unsigned int>& seen_keypoints_set) const {
        for(const auto& el: this->_points2keypoints) {
            if(el != -1) seen_keypoints_set.insert(el);
        }
    }


} // namespace SLucAM