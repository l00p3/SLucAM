//
// SLucAM_matcher.h
//
// This module describe the class useful to match points between two
// measurements in a flexible and general way.
//


#ifndef SLUCAM_MATCHER_H
#define SLUCAM_MATCHER_H

// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <opencv2/features2d.hpp>
#include <SLucAM_measurement.h>
#include <SLucAM_keypoint.h>



// -----------------------------------------------------------------------------
// Matcher class
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * This class represent a matcher that we need to match points between
    * two images. 
    */
    class Matcher {

    public:

        Matcher(const std::string& feat_types);

        void match_measurements(const Measurement& meas1, \
                                const Measurement& meas2,  
                                std::vector<cv::DMatch>& matches, \
                                const bool& match_th_high=true);

        static float compute_descriptors_distance(const cv::Mat& d1, \
                                                const cv::Mat& d2);
        
        static float compute_descriptors_distance(const cv::Mat& d1, \
                                                const std::vector<cv::Mat>& d2_set);

        static float get_match_th_max() { return Matcher::_match_th_max; };

    private:

        cv::Ptr<cv::BFMatcher> _bf_matcher;
        inline static bool _l2norm_dist = false;
        inline static float _match_th_high;     // Use this when you can handle outliers
        inline static float _match_th_low;      // Use this otherwise
        inline static float _match_th_max;      // Use this to have a max threshold for descriptors matching

    }; // class Matcher

} // namespace SLucAM



#endif // SLUCAM_MATCHER_H