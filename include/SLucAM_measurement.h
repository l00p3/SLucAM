//
// SLucAM_measurement.h
//
// In this module we have the class to deal with measurements.
//


#ifndef SLUCAM_MEASUREMENT_H
#define SLUCAM_MEASUREMENT_H

// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <opencv2/features2d.hpp>
#include <set>



// -----------------------------------------------------------------------------
// Measurement class
// -----------------------------------------------------------------------------
namespace SLucAM {
    
    /*
    * This class represent a single measurement, so it mantains the features 
    * extracted from an image.
    */
    class Measurement {
    
    public:
        
        Measurement(std::vector<cv::KeyPoint>& points, \
                    cv::Mat& descriptors);

        const std::vector<cv::KeyPoint>& getPoints() const \
                {return this->_points;};

        const cv::Mat& getDescriptors() const \
                {return this->_descriptors;};

        cv::Mat getDescriptor(const unsigned int& idx) const;

        unsigned int getAssociation(const unsigned int& p_idx) const \
                {return this->_points2keypoints[p_idx];};

        void getPointsAssociations(std::vector<std::pair<unsigned int, unsigned int>>& \
                                        points_associations) const;
        
        void getCommonKeypoints(const Measurement& meas2, \
                                std::vector<unsigned int>& commond_keypoints_ids) const;

        void addAssociation(const unsigned int& p_2d_idx, \
                                const unsigned int& p_3d_idx) \
                {this->_points2keypoints[p_2d_idx] = p_3d_idx;};
        
        void addAssociations(const std::vector<std::pair<unsigned int, unsigned int>>& \
                                        new_points_associations);

        void getObservedPoints(std::vector<unsigned int>& ids) const;

        void getObservedPointsSet(std::set<unsigned int>& seen_keypoints_set) const;

        bool isKeyframe() const {return this->_keyframe_idx!=-1;}

        void setKeyframeIdx(const int& idx) {this->_keyframe_idx = idx;}

        const int& getKeyframeIdx() const {return this->_keyframe_idx;}

        const unsigned int getId() const \
                {return this->_meas_id;};
        
        const std::string getImgName() const \
                {return this->_img_filename;};

        const void setImgName(const std::string& img_filename) \
                {this->_img_filename = img_filename;};

        // Take note of the next measurement id to use
        static unsigned int _next_id;

    private:

        // The set of points in the image
        std::vector<cv::KeyPoint> _points;

        // The set of descriptors, one per point in the image
        cv::Mat _descriptors;

        // The set of 3D points associated at each 2D point
        // (-1 is no such association exists)
        std::vector<unsigned int> _points2keypoints;

        // Idx of the keyframe for this measurement,
        // if it is not a keyframe, this value is -1
        int _keyframe_idx;

        // The id of the current measurement
        unsigned int _meas_id;

        // The name of the image which this measurement refers to,
        // if any
        std::string _img_filename;

    }; // class Measurement

} // namespace SLucAM



#endif // SLUCAM_MEASUREMENT_H