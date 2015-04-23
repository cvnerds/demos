/*
 * Homography image replacement demo.
 */



// std libs
#include <iostream>
#include <string>
using namespace std;


#include <cfloat>       // defines FLT_EPSILON

// custom libs
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
// using namespace cv;  // predilection









/********************************************************************************
 * @TODO: A quick and dirty logger
 ********************************************************************************/
extern int log_level;
int log_level = 0;
#define LOG_INFO(msg)           if (log_level > 0) { std::cout << " INFO :\t" << msg << std::endl; }
#define LOG_DEBUG(msg)          if (log_level > 1) { std::cout << " DEBUG:\t" << msg << std::endl; }
#define LOG_ERROR(msg)          std::cerr << " ERROR:\t" << msg << std::endl;
#define LOG_WARN(msg)           std::cerr << " WARN :\t" << msg << std::endl;






// C-like string formatting
// src: http://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf
template <typename... Ts>
std::string fmt(const std::string &fmt, Ts... vs)
{
    char b;
    unsigned required = std::snprintf(&b, 0, fmt.c_str(), vs...) + 1;
        // See comments: the +1 is necessary, while the first parameter
        //               can also be set to nullptr

    char bytes[required];
    std::snprintf(bytes, required, fmt.c_str(), vs...);

    return std::string(bytes);
}






/********************************************************************************
 * opencv utility functions
 ********************************************************************************/


namespace util_cv
{

/**
 * Load image located at given path
 */
cv::Mat load_image(const string& path, int options = CV_LOAD_IMAGE_COLOR)
{
    LOG_DEBUG("Loading " << path);
    cv::Mat mat = cv::imread(path, options);
    if (mat.empty())
    {
        LOG_ERROR("Image empty for file: " << path);
    }
    return mat;
}



/**
  Scales an image then displays it using OpenCV's inbuilt methods.

    :param img: input image
    :param title: window title
    :param wait: waitKey?
    :param windowScale: scale image
 */
void display_image(const cv::Mat& img, const string& title = "Debug", const bool wait = true, const float windowScale = 1.0, const bool close = false, const bool createWindow = true)
{
    cv::Mat tmp = img;
    if (windowScale != 1.0)
    {
        int height = int(round(img.rows * windowScale));
        int width  = int(round(img.cols * windowScale));
        cv::resize(img, tmp, cv::Size(width, height));
    }
    
    if (createWindow)
    {
        cv::namedWindow(title);
    }
    cv::imshow(title, tmp);
    if (wait)
    {
        cv::waitKey();
    }
    if (close)
    {
        cv::destroyWindow(title);
    }
}



/**
  Stacks a set of images horizontally before displaying

    :param img: input image
    :param title: window title
    :param wait: waitKey?
    :param windowScale: scale image
 */
void display_images(const vector<cv::Mat>& images, const string title = "Debug", const bool wait = true, const float windowScale = 1.0, const bool close = false, const bool createWindow = true)
{
    // @TODO support non-equally-sized images by padding zeros

    if (images.size() == 0)
    {
        return;
    }

    // stack images horizontally
    auto it_end = end(images);
    auto it = begin(images);
    cv::Mat tmp = (*it++);
    while (it != it_end)
    {
        cv::hconcat(tmp, (*it++), tmp);
    }
    display_image(tmp, title, wait, windowScale, close, createWindow);
}




void displayKeypoints(const cv::Mat& full_img, const vector<cv::KeyPoint>& keypoints)
{
    cv::namedWindow("keypoints", 1);
    cv::Mat img_keypoints;
    cv::drawKeypoints(full_img, keypoints, img_keypoints);
    cv::imshow("keypoints", img_keypoints);
    cv::waitKey(0);
}


void displayMatches(const cv::Mat &img_1, const std::vector<cv::KeyPoint> &keypoints_1, const cv::Mat &img_2, const std::vector<cv::KeyPoint> &keypoints_2, const std::vector<cv::DMatch> &matches, cv::Mat &img_matches)
{
    //-- Draw matches
    cv::drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches );
    //-- Show detected matches
    cv::namedWindow( "Matches", CV_WINDOW_NORMAL );
    cv::imshow("Matches", img_matches );
    cv::waitKey(0);
}




void draw_homography(const cv::Mat& img1, const cv::Mat& H, const cv::Mat &img2)
{
    int h = img1.rows;
    int w = img1.cols;
    
    std::vector<cv::Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(   0, 0);
    obj_corners[1] = cvPoint( w-1, 0 );
    obj_corners[2] = cvPoint( w-1, h-1 );
    obj_corners[3] = cvPoint(   0, h-1 );
    std::vector<cv::Point2f> scene_corners(4);
    std::vector<cv::Point2i> scene_corners_int;
    cv::perspectiveTransform(obj_corners, scene_corners, H);

    for (auto pt : scene_corners)
    {
        scene_corners_int.push_back( cv::Point2i(pt.x, pt.y) );
    }

    copy(begin(scene_corners), end(scene_corners), inserter(scene_corners_int, end(scene_corners_int)));
    cv::polylines(img2, scene_corners_int, true, cv::Scalar(0, 255, 0), 3, CV_AA);
}



};






/********************************************************************************
 * Datastructures
 ********************************************************************************/




/**
 * Data structure for a video frame
 */
class Frame
{
public:
    Frame(int index_, string name_)
    :
    index(index_),
    name(name_)
    {
    }

    // unique image ID, such as frame number
    int     index;
    // a possible user-defined name
    string  name;

    // the image data in BGR
    cv::Mat image;
};


/**
 * Datastructure for a video frame and (temporary) derived data
 */
class ProcessedFrame : public Frame
{
public:
    ProcessedFrame(int index, string name)
    :
    Frame(index,name)
    {
    }
    
    // the frame image as grayscale
    cv::Mat image_gray;
    // the final output image where the detected planar reference object is replaced with the replacement image
    cv::Mat image_replacement;

    // vector of 2d interest point locations
    vector<cv::KeyPoint> keypoints;
    // vector of feature descriptors for each interest point
    cv::Mat descriptors;
    // vector of feature matches
    vector<struct cv::DMatch> matches;
    // homography for the detected reference object
    cv::Mat homography;
};








/********************************************************************************
 * Main Demo Code
 ********************************************************************************/




/**
    Homography image replacement demo.

    This demo expects as input a folder with an image sequence.
    A reference image and a replacement image of the same size need to be provided (default: reference.jpg, replacement.jpg).
    This is a feature-based image matching demo that works for a planar object.
    The object (reference card) in the video is replaced by the replacement image.
    Think augmented reality.
 */
class DemoApp
{

public:
    /********************************************************************************
     * Datastructures
     ********************************************************************************/

    enum class MatcherType
    {
        BRUTEFORCE,
        FLANN
    };
    enum class DetectorType
    {
        SIFT,
        SURF,
        ORB,
        FAST,
        CUSTOM
    };
    enum class FeaturesType
    {
        SIFT,
        SURF,
        ORB,
        CUSTOM
    };

    typedef struct t_options
    {
        // verbosity count
        int verbose = 0;
        // toggle debuggin mode
        bool debug = false;

        // output directory for results
        string output = "./output";

        // reference image (planar object that is to be detected)
        string reference = "reference.jpg";
        // replacement image for the reference image in the scene
        string replacement = "replacement.jpg";

        // support different feature detectors, descriptors and matchers
        DetectorType detector_type = DetectorType::SURF;
        FeaturesType features_type = FeaturesType::SURF;
        MatcherType matcher_type   = MatcherType::FLANN;
    } Options;



private:
    /********************************************************************************
     * Class members
     ********************************************************************************/

    // holds all options of this demo
    Options _options;

    // list all frames from stdin
    vector<string> _paths;

    
    cv::Mat _im_reference;
    cv::Mat _im_replacemt;
    cv::Mat _im_reference_gray;
    cv::Mat _im_replacemt_gray;

    cv::Ptr<cv::FeatureDetector>        _detector;
    cv::Ptr<cv::DescriptorExtractor>    _extractor;
    cv::Ptr<cv::DescriptorMatcher>      _matcher;

    vector<cv::KeyPoint> _keypoints_ref;
    cv::Mat _descriptors_ref;


public:

    /********************************************************************************
     * Logic
     ********************************************************************************/
    DemoApp()
    {
    }


    DemoApp(Options& options)
    :
    _options(options)
    {
        // empty function
    }

    

public:

    /********************************************************************************
     * Public API
     ********************************************************************************/

    // Main entry point of the app
    void run()
    {
        /* some initial setup */

        // list all frames from stdin
        _paths = listSequence();

        // load reference images
        _im_reference = util_cv::load_image(_options.reference);
        _im_replacemt = util_cv::load_image(_options.replacement);
        cv::cvtColor(_im_reference, _im_reference_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(_im_replacemt, _im_replacemt_gray, cv::COLOR_BGR2GRAY);

        if (_options.debug)
        {
            util_cv::display_images(vector<cv::Mat>( {_im_reference, _im_replacemt} ), "Reference/Replacement");
        }

        
        // initialise modules
        _detector  = initFeatureDetector();
        _extractor = initFeatureExtractor();
        _matcher   = initFeatureMatcher();

        
        /* pre-compute features for referenc images */
        _keypoints_ref = detectKeypoints(_im_reference_gray);
        _descriptors_ref = computeFeatures(_im_reference_gray, _keypoints_ref);
        

        /* main loop (sequential algorithm) */
        runLoop(_paths);
    }


    // Process input frame-by-frame
    void runLoop(vector<string> &paths)
    {
        LOG_INFO(" * Analysing image sequence");
        
        const int N = paths.size();

        if (N == 0)
        {
            LOG_WARN(" Empty sequence, nothing to be done");
            return;
        }
        
        int frame_count = -1;
        for (const auto& path : paths)
        {
            ++frame_count;


            LOG_INFO("");
            LOG_INFO(" * frame " << frame_count << " : " << path);

            // set up frame object
            ProcessedFrame frame(frame_count, path);

            // load frame image
            frame.image = util_cv::load_image(path);
            cv::cvtColor(frame.image, frame.image_gray, cv::COLOR_BGR2GRAY);

            // detect and compute features
            frame.keypoints   = detectKeypoints(frame.image_gray);
            frame.descriptors = computeFeatures(frame.image_gray, frame.keypoints);
            
            // match feature descriptors
            frame.matches = matchFeatures(frame.descriptors);

            // debug features
            if (_options.debug)
            {
                cv::Mat debugImage;
                cv::drawMatches(_im_reference, _keypoints_ref, frame.image, frame.keypoints, frame.matches, debugImage);
                util_cv::display_image(debugImage, "DEBUG");
            }

            // compute homography from the reference object to the scene
            frame.homography = computeHomography(_keypoints_ref, frame.keypoints, frame.matches);


            // debug homography
            if (_options.debug)
            {
                cv::Mat debugImage = frame.image.clone();
                frame.homography.convertTo(frame.homography, CV_32FC1);
                util_cv::draw_homography(_im_reference, frame.homography, debugImage);
                cv::drawMatches(_im_reference, _keypoints_ref, debugImage.clone(), frame.keypoints, frame.matches, debugImage);
                util_cv::display_image(debugImage, "DEBUG");
            }

            // check if homography is bad
            bool bad_homography = false;
            // If the det of the upper 2x2 matrix of the homography is smaller than 0, it is not orientation-preserving.
            // src: http://answers.opencv.org/question/2588/check-if-homography-is-good/
            // This filters out some bad frames, although it's not a perfect criteria (tested with the bad results using BruteForceMatcher)
            double det2x2 = cv::determinant( frame.homography(cv::Rect(0,0,2,2)).clone() );
            if (det2x2 < 0)
            {
                bad_homography = true;
            }

            // This criteria can be found in the image stitching pipeline of opencv...
            double det3x3 = abs( cv::determinant(frame.homography) );
            if (det3x3 < 0.00001) // alternative: FLT_EPSILON
            {
                bad_homography = true;
            }

            if (bad_homography)
            {
                // homography is useless, just show the original picture
                frame.image_replacement = frame.image.clone();
            }
            else
            {
                // warp replacement image and merge with the frame
                int h = frame.image.rows;
                int w = frame.image.cols;
                cv::Mat im_warped;
                cv::warpPerspective(_im_replacemt, im_warped, frame.homography, cv::Size(w, h));

                frame.image_replacement = frame.image.clone();
                im_warped.copyTo(frame.image_replacement, im_warped > 0);
            }


            // debug result
            if (_options.debug)
            {
                cv::Scalar color = cv::Scalar(255, 0, 0);
                if (det2x2 < 0 || det3x3 < 0.00001) // alt: FLT_EPSILON
                {
                    color = cv::Scalar(0, 0, 255);
                }

                cv::putText(frame.image_replacement, fmt("Frame %d: %s", frame_count, path.c_str()), cv::Point2i(0,25), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0));
                cv::putText(frame.image_replacement, fmt("det2x2 = %f", det2x2), cv::Point2i(0,50), cv::FONT_HERSHEY_SIMPLEX, 0.5, color);
                cv::putText(frame.image_replacement, fmt("det3x3 = %f", det3x3), cv::Point2i(0,75), cv::FONT_HERSHEY_SIMPLEX, 0.5, color);
                util_cv::display_images(vector<cv::Mat>( {frame.image, frame.image_replacement} ), "Processed Frame");
            }

            // write image
            if (_options.output.length() != 0)
            {
                string output = fmt("%s/%s", _options.output.c_str(), path.substr( path.find_last_of('/')+1 ).c_str());
                LOG_INFO( " * Write " << output );
                cv::imwrite( output, frame.image_replacement );
            }
        }
    }



    /*
     * Parse STDIN and assume each line contains a file path
     */
    vector<string> listSequence()
    {
        LOG_INFO(" * Read input sequence from STDIN ");

        vector<string> files;
        files.reserve(50);
        string line;
        while (std::getline(std::cin, line))
        {
            files.push_back(line);
        }

        LOG_INFO(" * Found " << files.size() << " files");

        return files;
    }


    /*
     * Initialise keypoint detector
     */
    cv::Ptr<cv::FeatureDetector> initFeatureDetector()
    {
        cv::Ptr<cv::FeatureDetector> featureDetector;
        
        switch (_options.detector_type)
        {
        case DetectorType::ORB:
            featureDetector = cv::FeatureDetector::create("PyramidORB");
            break;
        case DetectorType::FAST:
            featureDetector = cv::FeatureDetector::create("PyramidFAST");
            break;
        case DetectorType::CUSTOM:
            featureDetector = new cv::PyramidAdaptedFeatureDetector(
                new cv::DynamicAdaptedFeatureDetector(
                    // new cv::FastAdjuster(20, true), 30, 50, 20),
                   new cv::FastAdjuster(20, true), 5000, 10000, 10),
                    8
                );
            break;
        case DetectorType::SIFT:
            featureDetector = cv::FeatureDetector::create("SIFT");
            break;
        case DetectorType::SURF: // default
        default:
            featureDetector = cv::FeatureDetector::create("PyramidSURF");
            break;
        }

        return featureDetector;
    }


    /*
     * Initialise descriptor extractor
     */
    cv::Ptr<cv::DescriptorExtractor> initFeatureExtractor()
    {
        cv::Ptr<cv::DescriptorExtractor> extractor;

        switch (_options.features_type)
        {
        case FeaturesType::ORB:
            extractor = cv::DescriptorExtractor::create("ORB");
            break;
        case FeaturesType::SIFT:
            extractor = cv::DescriptorExtractor::create("SIFT");
            break;

        case FeaturesType::CUSTOM: // TODO
        case FeaturesType::SURF: // default
        default:
            extractor = cv::DescriptorExtractor::create("SURF");
            break;
        }

        return extractor;
    }


    /*
     * Initialise the feature matcher
     */
    cv::Ptr<cv::DescriptorMatcher> initFeatureMatcher()
    {
        cv::Ptr<cv::DescriptorMatcher> matcher;
        if (_options.matcher_type == MatcherType::BRUTEFORCE)
        {
            const int normType = cv::NORM_L2;

            matcher = new cv::BFMatcher(normType);
        }
        else
        {
            const cv::Ptr<cv::flann::IndexParams>& indexParams = new cv::flann::KDTreeIndexParams();
            const cv::Ptr<cv::flann::SearchParams>& searchParams = new cv::flann::SearchParams();

            matcher = new cv::FlannBasedMatcher(indexParams, searchParams);
        }
        return matcher;
    }



    /*
     * Detect interesting keypoints
     */
    vector<cv::KeyPoint> detectKeypoints(const cv::Mat& image)
    {
        vector<cv::KeyPoint> keypoints;
        (*_detector).detect(image, keypoints);
        return keypoints;
    }



    /*
     * Describe interesting keypoints
     */
    cv::Mat computeFeatures(const cv::Mat& image, vector<cv::KeyPoint>& keypoints)
    {
        cv::Mat descriptors;
        (*_extractor).compute(image, keypoints, descriptors);
        return descriptors;
    }



    /*
     * Match feature vectors against reference features
     */
    vector<struct cv::DMatch> matchFeatures(const cv::Mat& features_query)
    {
        vector< vector<struct cv::DMatch> > matches;
        vector<struct cv::DMatch> good_matches;
        if (_options.matcher_type == MatcherType::FLANN)
        {
            // bug: flann enums are missing
            (*_matcher).knnMatch(_descriptors_ref, features_query, matches, 2);
            // store all the good matches as per Lowe's ratio test.
            good_matches = applyRatioTest(matches);
        }
        else
        {
            (*_matcher).match(_descriptors_ref, features_query, good_matches);
        }

        return good_matches;
    }




    /*
     * Ratio test from D.Lowe's SIFT paper
     */
    vector<struct cv::DMatch> applyRatioTest(vector< vector<struct cv::DMatch> >& matches, const float& ratio = 0.75)
    {
        /*
            An alternate method of determining high-quality feature matches is the ratio test
            proposed by David Lowe in his paper on SIFT (page 20 for an explanation).
            This test rejects poor matches by computing the ratio between the best and
            second-best match. If the ratio is below some threshold, the match is discarded as being low-quality.
            
            src: http://stackoverflow.com/questions/17967950/improve-matching-of-feature-points-with-opencv
        */

        vector<struct cv::DMatch> good;
        good.reserve(matches.size());
        int i = 0;
        for (auto v_match : matches)
        {
            if (v_match[0].distance < ratio * v_match[1].distance)
            {
                good.push_back(v_match[0]);
            }
        }
        
        return good;
    }



    /*
     * Compute homography given keypoints and matches
     */
    cv::Mat computeHomography(const vector<cv::KeyPoint>& keypoints_src, const vector<cv::KeyPoint>& keypoints_dst, const vector<struct cv::DMatch>& good_matches)
    {
        const int N = good_matches.size();

        cv::vector<cv::Point2f> src_pts;
        cv::vector<cv::Point2f> dst_pts;
        src_pts.reserve(N);
        dst_pts.reserve(N);
        for (auto match : good_matches)
        {
            src_pts.push_back( keypoints_src[match.queryIdx].pt );
            dst_pts.push_back( keypoints_dst[match.trainIdx].pt );
        }

        cv::Mat mask;
        cv::Mat homography = cv::findHomography( src_pts, dst_pts, mask, CV_RANSAC, 5.0);

#if 0
        // TODO: This two-stage optimisation is not really better (seq_00000071.jpg).. why?..

        cv::vector<cv::Point2f> src_inliers;
        cv::vector<cv::Point2f> dst_inliers;
        src_inliers.reserve(N);
        dst_inliers.reserve(N);
        
        for (int i=0; i<mask.rows; ++i)
        {
            auto inlier = mask.at<double>(i,0);
            if (inlier > 0)
            {
                src_inliers.push_back( src_pts[i] );
                dst_inliers.push_back( dst_pts[i] );
            }
        }

        homography = cv::findHomography( src_inliers, dst_inliers, CV_LMEDS);
#endif

        LOG_DEBUG("H=" << homography);

        return homography;
    }
};




int main(int argc, char** argv)
{
    // initialize the app (contains default values for some command line arguments)
    log_level = 2;
    DemoApp::Options options;
    options.debug = true;
    DemoApp app(options);
    app.run();
}
