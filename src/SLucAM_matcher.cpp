//
// SLucAM_matcher.h implementation
//


// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <SLucAM_matcher.h>
#include <limits>

// TODO: delete this
#include <iostream>
using namespace std;




// -----------------------------------------------------------------------------
// Implementation of the Matcher class methods
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * Default constructor of the Matcher class. If we use ORB features we use
    * NORM_HAMMING distance for matching, otherwise we use NORM_L2.
    */
    Matcher::Matcher(const std::string& feat_types) {
        if(feat_types == "orb") {
            this->_bf_matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
            Matcher::_l2norm_dist = false;
            Matcher::_match_th_high = 30;
            Matcher::_match_th_low = 10;
            Matcher::_match_th_max = 50;
        } else {
            this->_bf_matcher = cv::BFMatcher::create(cv::NORM_L2, true);
            Matcher::_l2norm_dist = true;
            if(feat_types == "lf_net") {
                Matcher::_match_th_high = 0.5;
            } else {
                Matcher::_match_th_high = 0.7;
            }
            Matcher::_match_th_low = 0.2;
            Matcher::_match_th_max = 1.0;
        }
    }


    /*
    * Function that, given two measurement, determime which keypoints are
    * the same between the two.
    * Inputs:
    *   meas1/meas2: the two measurements to match
    *   matches: vector where to store the matches
    */
   void Matcher::match_measurements(const Measurement& meas1, \
                                const Measurement& meas2,  
                                std::vector<cv::DMatch>& matches, \
                                const bool& match_th_high) {
    
        std::vector<cv::DMatch> unfiltered_matches;
        this->_bf_matcher->match(meas1.getDescriptors(), \
                                meas2.getDescriptors(), \
                                unfiltered_matches);

        // Adjust the threshold if we use L2 NORM
        float th;
        if(match_th_high)
            th = Matcher::_match_th_high;
        else
            th = Matcher::_match_th_low;

        // Filter matches
        matches.reserve(unfiltered_matches.size());
        for(auto& m : unfiltered_matches) {
            //std::cout << "DIST: " << m.distance << std::endl;
            if(m.distance <= th) 
                matches.emplace_back(m);
        }
        matches.shrink_to_fit();

        return;

    }



    /*
    * This function, given two ORB descriptors (d1, d2) computes the distance
    * between them using the bit set count operation from:
    * http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
    */
    float Matcher::compute_descriptors_distance(const cv::Mat& d1, \
                                                const cv::Mat& d2) {
    
        float distance;

        if(Matcher::_l2norm_dist) {
            distance = (float)cv::norm(d1, d2, cv::NORM_L2);
        } else {

            const int *d1_ptr = d1.ptr<int32_t>();
            const int *d2_ptr = d2.ptr<int32_t>();

            distance = 0;

            for(int i=0; i<8; i++, d1_ptr++, d2_ptr++)
            {
                unsigned  int v = *d1_ptr ^ *d2_ptr;
                v = v - ((v >> 1) & 0x55555555);
                v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
                distance += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
            }

        }

        return (float)distance;

    }



    /*
    * This function computes the distance between a single descriptor and a set of
    * descriptors and returns the nearest distance computed.
    */
    float Matcher::compute_descriptors_distance(const cv::Mat& d1, \
                                                const std::vector<cv::Mat>& d2_set) {
    
        // Initialization
        float best_distance = std::numeric_limits<float>::max();
        float current_distance;
        const unsigned int n_descriptors = d2_set.size();

        // Search for the best distance
        for(unsigned int i=0; i<n_descriptors; ++i) {
            current_distance = compute_descriptors_distance(d1, d2_set[i]);
            if(current_distance < best_distance) 
                best_distance = current_distance;
        }

        return best_distance;
    }

} // namespace SLucAM