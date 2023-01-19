//
// SLucAM_image.h
//
// In this module are defined all the functions to deal with images, included
// functions to extract features from them and find associations between 
// keypoints.
//


#ifndef SLUCAM_IMAGE_H
#define SLUCAM_IMAGE_H

// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <string>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <SLucAM_measurement.h>



// -----------------------------------------------------------------------------
// Feature Extractor class
// -----------------------------------------------------------------------------
namespace SLucAM {

    class FeatureExtractor {

    public:

        FeatureExtractor(bool ANMS=true, \
                        unsigned int fast_threshold=7, \
                        unsigned int ANMS_points=2000, \
                        float tolerance=0.1);

        void extract_features(const cv::Mat& img, \
                                std::vector<cv::KeyPoint>& keypoints, \
                                cv::Mat& descriptors);

    private:

        // ORB feature extractor
        cv::Ptr<cv::ORB> _orb_extractor;

        // FAST features detector
        cv::Ptr<cv::FastFeatureDetector> _detector;

        // Extractor of FAST descriptors
        cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> _extractor;

        // Fast threshold
        unsigned int _fast_threshold;

        // Number of points to mantain with ANMS
        unsigned int _ANMS_points;

        // Tolerance of the number of return points
        float _tolerance;

        // Flag if to use ANMS (true) or ORB (false) for feature extraction
        bool _ANMS;

    };

} // namespace SLucAM



// -----------------------------------------------------------------------------
// Loading and saving utilities
// -----------------------------------------------------------------------------
namespace SLucAM {
    bool load_image(const std::string& filename, cv::Mat& img);
} // namespace SLucAM



// -----------------------------------------------------------------------------
// Visualization utilities
// -----------------------------------------------------------------------------
namespace SLucAM {
    bool visualize_matches(const Measurement& meas1, const Measurement& meas2, \
                        const std::vector<cv::DMatch>& matches, \
                        const std::vector<unsigned int>& matches_filter);
} // namespace SLucAM


#endif // SLUCAM_IMAGE_H