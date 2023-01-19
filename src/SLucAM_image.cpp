//
// SLucAM_image.h implementation
//


// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <SLucAM_image.h>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <numeric>
#include <anms.h>



// -----------------------------------------------------------------------------
// Implementation of Feature Extractor class methods
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * Main constructor, it initializes all the parameters. In particular
    * if the ANMS flag is set to false, then ORB Feature Extractor will
    * be used, so we do not need the other parameters to be specified.
    */
    FeatureExtractor::FeatureExtractor(bool ANMS, \
                                        unsigned int fast_threshold, \
                                        unsigned int ANMS_points, \
                                        float tolerance) {

        this->_ANMS = ANMS;

        if(ANMS) {
            this->_fast_threshold = fast_threshold;
            this->_ANMS_points = ANMS_points;
            this->_tolerance = tolerance;

            this->_detector = cv::FastFeatureDetector::create(this->_fast_threshold, true);
            this->_extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(32);
        } else {
            this->_orb_extractor = cv::ORB::create(1000);
        }
        
    }

    void FeatureExtractor::extract_features(const cv::Mat& img, \
                                            std::vector<cv::KeyPoint>& keypoints, \
                                            cv::Mat& descriptors) {
        
        // ANMS ferature extractor
        if(this->_ANMS) {
            // Initialization
            std::vector<float> response_vec;

            // Extract FAST features
            this->_detector->detect(img, keypoints);
            const unsigned int n_keypoints = keypoints.size();

            // Sort keypoints by decreasing order of strength
            for (unsigned int i = 0; i < n_keypoints; i++)
                response_vec.push_back(keypoints[i].response);
            std::vector<int> idxs(response_vec.size());
            std::iota(std::begin(idxs), std::end(idxs), 0);
            cv::sortIdx(response_vec, idxs, cv::SORT_DESCENDING);
            std::vector<cv::KeyPoint> sorted_keypoints;
            for (unsigned int i = 0; i < n_keypoints; i++)
                sorted_keypoints.push_back(keypoints[idxs[i]]);
            
            // Apply SSC ANMS
            keypoints = ssc(sorted_keypoints, this->_ANMS_points, this->_tolerance, \
                                img.cols, img.rows);

            // Compute descriptors
            this->_extractor->compute(img, keypoints, descriptors);
        
        // ORB feature extractor
        } else {
            this->_orb_extractor->detectAndCompute(img, cv::noArray(), \
                                                    keypoints, descriptors);
        }

    }

} // namespace SLucAM



// -----------------------------------------------------------------------------
// Implementation of loading and saving utilities
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * Load an image, checking for the consistency of the inputs,
    * in case of errors it returns "false".
    * Inputs:
    *   filename: relative path of the image to load
    *   img: OpenCV Matrix object where the image will be loaded
    */
    bool load_image(const std::string& filename, cv::Mat& img) {
        img = cv::imread(filename, cv::IMREAD_UNCHANGED);
        if (img.empty()) {
            return false;
        }
        cv::cvtColor(img, img, cv::COLOR_RGB2GRAY);
        return true;
    }

} // namespace SLucAM



// -----------------------------------------------------------------------------
// Implementation of visualization utilities
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * Useful to visualize the matches between two images.
    */
    bool visualize_matches(const Measurement& meas1, const Measurement& meas2, \
                        const std::vector<cv::DMatch>& matches, \
                        const std::vector<unsigned int>& matches_filter) {
        
        // Get the names of the imgs to visualize
        const std::string& img1_filename = meas1.getImgName();
        const std::string& img2_filename = meas2.getImgName();

        // Load the images
        cv::Mat img1, img2;
        if(!load_image(img1_filename, img1)) return false;
        if(!load_image(img2_filename, img2)) return false;

        // Filter out the matches
        const unsigned int n_matches = matches_filter.size();
        std::vector<cv::DMatch> filtered_matches;
        filtered_matches.reserve(n_matches);
        for(unsigned int i=0; i<n_matches; ++i) {
            filtered_matches.emplace_back(matches[matches_filter[i]]);
        }
        filtered_matches.shrink_to_fit();

        cv::Mat img_matches;
        cv::drawMatches(img1, meas1.getPoints(), img2, meas2.getPoints(),
                    filtered_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::namedWindow("MATCHED IMGS", cv::WINDOW_AUTOSIZE);
        cv::imshow("MATCHED IMGS", img_matches);
        cv::waitKey(0);
        cv::destroyAllWindows();

        return true;
    }

} // namespace SLucAM