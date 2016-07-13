///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2016, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// THIS SOFTWARE IS PROVIDED ๏ฟฝAS IS๏ฟฝ FOR ACADEMIC USE ONLY AND ANY EXPRESS
// OR IMPLIED WARRANTIES WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
// BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY.
// OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Notwithstanding the license granted herein, Licensee acknowledges that certain components
// of the Software may be covered by so-called ๏ฟฝopen source๏ฟฝ software licenses (๏ฟฝOpen Source
// Components๏ฟฝ), which means any software licenses approved as open source licenses by the
// Open Source Initiative or any substantially similar licenses, including without limitation any
// license that, as a condition of distribution of the software licensed under such license,
// requires that the distributor make the software available in source code format. Licensor shall
// provide a list of Open Source Components for a particular version of the Software upon
// Licensee๏ฟฝs request. Licensee will comply with the applicable terms of such licenses and to
// the extent required by the licenses covering Open Source Components, the terms of such
// licenses will apply in lieu of the terms of this Agreement. To the extent the terms of the
// licenses applicable to Open Source Components prohibit any of the restrictions in this
// License Agreement with respect to such Open Source Component, such restrictions will not
// apply to such Open Source Component. To the extent the terms of the licenses applicable to
// Open Source Components require Licensor to make an offer to provide source code or
// related information in connection with the Software, such offer is hereby made. Any request
// for source code or related information should be directed to cl-face-tracker-distribution@lists.cam.ac.uk
// Licensee acknowledges receipt of notices for the Open Source Components for the initial
// delivery of the Software.

//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace: an open source facial behavior analysis toolkit
//       Tadas Baltru๏ฟฝaitis, Peter Robinson, and Louis-Philippe Morency
//       in IEEE Winter Conference on Applications of Computer Vision, 2016
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltru๏ฟฝaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling
//       in IEEE International. Conference on Computer Vision (ICCV),  2015
//
//       Cross-dataset learning and person-speci?c normalisation for automatic Action Unit detection
//       Tadas Baltru๏ฟฝaitis, Marwa Mahmoud, and Peter Robinson
//       in Facial Expression Recognition and Analysis Challenge,
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015
//
//       Constrained Local Neural Fields for robust facial landmark detection in the wild.
//       Tadas Baltru๏ฟฝaitis, Peter Robinson, and Louis-Philippe Morency.
//       in IEEE Int. Conference on Computer Vision Workshops, 300 Faces in-the-Wild Challenge, 2013.
//
///////////////////////////////////////////////////////////////////////////////
// FaceTrackingVid.cpp : Defines the entry point for the console application for tracking faces in videos.

// Libraries for landmark detection (includes CLNF and CLM modules)
#include "LandmarkCoreIncludes.h"
#include "GazeEstimation.h"

#include <fstream>
#include <sstream>

// OpenCV includes
#include <opencv2/videoio/videoio.hpp>  // Video write
#include <opencv2/videoio/videoio_c.h>  // Video write
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Boost includes
#include <filesystem.hpp>
#include <filesystem/fstream.hpp>

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

static void printErrorAndAbort( const std::string & error )
{
    std::cout << error << std::endl;
    abort();
}

#define FATAL_STREAM( stream ) \
printErrorAndAbort( std::string( "Fatal error: " ) + stream )

using namespace std;

vector<string> get_arguments(int argc, char **argv)
{

    vector<string> arguments;

    for(int i = 0; i < argc; ++i)
    {
        arguments.push_back(string(argv[i]));
    }
    return arguments;
}

///ไบบ็ผ็ถๆ€ๅคๆ–ญ็ธๅ…ณ
std::vector<double> mVecEyeL;		//ๅทฆ็ผ็ถๆ€ๆฏ”ๅ€ผๅ‘้๏ผๅฝ“ๅๅ€ผไธบๆ€ๆๅ‘้ๅ…็ด ็ๅ ๆๅนณๅ๏ผ่ฟๆปคๆๅชๅฃฐ็น
std::vector<double> mVecEyeR;		//ๅณ็ผ็ถๆ€ๆฏ”ๅ€ผๅ‘้
int mEyeNumOfVec = 10;				//ๅ‘้ๅคงๅฐ

// Some globals for tracking timing information for visualisation
double fps_tracker = -1.0;
int64 t0 = 0;

// judge face pose and reset
void judgeFacePose(LandmarkDetector::CLNF& face_model,cv::Vec6d& pose_estimate)
{

    //face_model.Reset();

    // output face pose angle
    int yawAngle = (int)(pose_estimate[4]*180/CV_PI);
    int pitchAngle = (int)(pose_estimate[3]*180/CV_PI);
    int rollAngle = (int)(pose_estimate[5]*180/CV_PI);

    ///////////////////////////////////////
    /// Face pose estimation
    /// input  : camera param, face landmarks
    /// output : pitch yaw roll angle
    ///////////////////////////////////////

    // checkout pose state if pose isnt zero and pose same as next pose
    // compare face pose between two frame
    if((face_model.poseAngle[0] == pitchAngle &&
        face_model.poseAngle[1] == yawAngle &&
        face_model.poseAngle[2] == rollAngle)
            && !(pitchAngle == 0 && yawAngle == 0 && rollAngle ==0))
    {
        face_model.angle_nochange_num += 1;

        if(face_model.angle_nochange_num >=25)
        {
            cout << face_model.angle_nochange_num<<endl;
            face_model.Reset();
            face_model.angle_nochange_num = 0;
        }
    }
    else
    {
        face_model.angle_nochange_num = 0;
    }

    // save previous face pose
    face_model.poseAngle[0] = pitchAngle;
    face_model.poseAngle[1] = yawAngle;
    face_model.poseAngle[2] = rollAngle;

}

// Visualising the results
void visualise_tracking(cv::Mat& captured_image, cv::Mat_<float>& depth_image, LandmarkDetector::CLNF& face_model, const LandmarkDetector::FaceModelParameters& det_parameters, int frame_count ,double fx, double fy,double cx,double cy)
{
    // Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
    double detection_certainty = face_model.detection_certainty;
    bool detection_success = face_model.detection_success;

    cv::Vec6d pose_estimate_to_draw;

    double visualisation_boundary = 0.2;
    // Only draw if the reliability is reasonable, the value is slightly ad-hoc
    if (detection_certainty < visualisation_boundary)
    {
        LandmarkDetector::Draw(captured_image, face_model);
        /*
        int n = face_model.detected_landmarks.rows / 2;
        ///-----------ๆๅ–ๅณ็ผ่ฟ่กๅๆ-----------///
        std::vector<cv::Point> pts;
        cv::Scalar eyeColor(255, 255, 0);
        for (int k = 42; k < 48; ++k)
        {
            cv::Point pt((int)face_model.detected_landmarks.at<double>(k), (int)face_model.detected_landmarks.at<double>(k+ n));
            pts.push_back(pt);
            cv::drawMarker(captured_image, pt, eyeColor, cv::MarkerTypes::MARKER_CROSS, 5, 1);
        }
        //ๆ€ๅฐๅค–ๆฅ็ฉๅฝข
        cv::RotatedRect rrt = cv::minAreaRect(pts);
        cv::Point2f pts4[4];
        rrt.points(pts4);
        //่ฎก็ฎ—็ผ็ๆ€ๅฐๅค–ๆฅ็ฉๅฝข็้•ฟๅฎฝๆฏ”
        double ratioR = cv::norm(pts4[0] - pts4[1]) / cv::norm(pts4[1] - pts4[2]);
        if (ratioR > 1)
            ratioR = 1.0 / ratioR;
        //ๅ ๅ…ฅๅ‘้่ฟ่กๆปคๆณขๅค็
        if (mVecEyeR.size() >= mEyeNumOfVec)
            mVecEyeR.erase(mVecEyeR.begin());
        mVecEyeR.push_back(ratioR);
        ratioR = cv::mean(mVecEyeR).val[0];
        //่พ“ๅบ้•ฟๅฎฝๆฏ”๏ผ่ฐ่ฏ•ไฟกๆฏ
        char c[30];
        sprintf(c, "%0.2lf", ratioR);
        cv::Point pt(rrt.boundingRect().tl());
        pt.y -= 10;
        cv::putText(captured_image, cv::String(c), pt, cv::FONT_ITALIC, 0.6, cv::Scalar(255, 0, 255), 2);

        ///ๆๅ–ๅทฆ็ผ่ฟ่กๅๆ
        pts.clear();
        eyeColor = cv::Scalar(0, 255, 0);
        for (int k = 36; k < 42; ++k)
        {
            cv::Point pt((int)face_model.detected_landmarks.at<double>(k), (int)face_model.detected_landmarks.at<double>(k+ n));
            pts.push_back(pt);
            cv::drawMarker(captured_image, pt, eyeColor, cv::MarkerTypes::MARKER_CROSS, 5, 1);
        }
        //ๆ€ๅฐๅค–ๆฅ็ฉๅฝข
        rrt = cv::minAreaRect(pts);
        rrt.points(pts4);
        //่ฎก็ฎ—็ผ็ๆ€ๅฐๅค–ๆฅ็ฉๅฝข็้•ฟๅฎฝๆฏ”
        double ratioL = cv::norm(pts4[0] - pts4[1]) / cv::norm(pts4[1] - pts4[2]);
        if (ratioL > 1)
            ratioL = 1.0 / ratioL;
        //ๅ ๅ…ฅๅ‘้่ฟ่กๆปคๆณขๅค็
        if (mVecEyeL.size() >= mEyeNumOfVec)
            mVecEyeL.erase(mVecEyeL.begin());
        mVecEyeL.push_back(ratioL);
        ratioL = cv::mean(mVecEyeL).val[0];
        sprintf(c, "%0.2lf", ratioL);
        pt = cv::Point(rrt.boundingRect().tl());
        pt.y -= 10;
        cv::putText(captured_image, cv::String(c), pt, cv::FONT_ITALIC, 0.6, cv::Scalar(255, 0, 255), 2);
        */
        double vis_certainty = detection_certainty;
        if (vis_certainty > 1)
            vis_certainty = 1;
        if (vis_certainty < -1)
            vis_certainty = -1;

        vis_certainty = (vis_certainty + 1) / (visualisation_boundary + 1);
        pose_estimate_to_draw = LandmarkDetector::GetCorrectedPoseWorld(face_model, fx, fy, cx, cy);
        judgeFacePose(face_model, pose_estimate_to_draw);


        // A rough heuristic for box around the face width
       // int thickness = (int)std::ceil(2.0* ((double)captured_image.cols) / 640.0);

        // Draw it in reddish if uncertain, blueish if certain
        //LandmarkDetector::DrawBox(captured_image, pose_estimate_to_draw, cv::Scalar((1 - vis_certainty)*255.0, 0, vis_certainty * 255), thickness, fx, fy, cx, cy);

        //if (det_parameters.track_gaze && detection_success && face_model.eye_model)
        //{
            //FaceAnalysis::DrawGaze(captured_image, face_model, gazeDirection0, gazeDirection1, fx, fy, cx, cy);
        //}
    }

    // Work out the framerate
    if (frame_count % 10 == 0)
    {
        double t1 = cv::getTickCount();
        fps_tracker = 10.0 / (double(t1 - t0) / cv::getTickFrequency());
        t0 = t1;
    }

    // Write out the framerate on the image before displaying it
    char fpsC[255];
    std::sprintf(fpsC, "%d", (int)fps_tracker);
    string fpsSt("FPS:");
    fpsSt += fpsC;


    // output face pose angle
    cv::putText(captured_image, fpsSt, cv::Point(10, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0),2);
    int yawAngle = (int)(pose_estimate_to_draw[4]*180/CV_PI);
    int pitchAngle = (int)(pose_estimate_to_draw[3]*180/CV_PI);
    int rollAngle = (int)(pose_estimate_to_draw[5]*180/CV_PI);

    if(yawAngle ==0 && pitchAngle ==0 && rollAngle ==0)
    {
        cv::putText(captured_image, "No people or angle overflow", cv::Point(250, 50), CV_FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(255 ,0, 0),2);
    }
    else
    {
        cv::putText(captured_image, "Pitch  : "+std::to_string(pitchAngle), cv::Point(10, 50),  CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0, 0, 255),2);
        cv::putText(captured_image, "Yaw   : " +std::to_string(yawAngle), cv::Point(10, 80),  CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0 ,0, 255),2);
        cv::putText(captured_image, "Roll   : "+std::to_string(rollAngle), cv::Point(10, 110), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0, 0, 255),2);
    }

    if (!det_parameters.quiet_mode)
    {
        cv::namedWindow("tracking_result", 1);
        cv::imshow("tracking_result", captured_image);

        if (!depth_image.empty())
        {
            // Division needed for visualisation purposes
            imshow("depth", depth_image / 2000.0);
        }
    }
}

int main (int argc, char **argv)
{

    //////////////////////////////
    /// Input arguments
    //////////////////////////////
    vector<string> arguments = get_arguments(argc, argv);

    // Some initial parameters that can be overriden from command line
    vector<string> files, depth_directories, output_video_files, out_dummy;

    // By default try webcam 0
    int device = 0;

    LandmarkDetector::FaceModelParameters det_parameters(arguments);
    
    // Get the input output file parameters

    // Indicates that rotation should be with respect to world or camera coordinates
    bool u;
    LandmarkDetector::get_video_input_output_params(files, depth_directories, out_dummy, output_video_files, u, arguments);

    // The modules that are being used for tracking
    LandmarkDetector::CLNF clnf_model(det_parameters.model_location);

    // Grab camera parameters, if they are not defined (approximate values will be used)
    float fx = 0, fy = 0, cx = 0, cy = 0;
    // Get camera parameters
    LandmarkDetector::get_camera_params(device, fx, fy, cx, cy, arguments);

    // If cx (optical axis centre) is undefined will use the image size/2 as an estimate
    bool cx_undefined = false;
    bool fx_undefined = false;
    if (cx == 0 || cy == 0)
    {
        cx_undefined = true;
    }
    if (fx == 0 || fy == 0)
    {
        fx_undefined = true;
    }

    // If multiple video files are tracked, use this to indicate if we are done
    bool done = false;
    int f_n = -1;

    det_parameters.track_gaze = false;

    while(!done) // this is not a for loop as we might also be reading from a webcam
    {

        string current_file;

        // We might specify multiple video files as arguments
        if(files.size() > 0)
        {
            f_n++;
            current_file = files[f_n];
        }
        else
        {
            // If we want to write out from webcam
            f_n = 0;
        }

        bool use_depth = !depth_directories.empty();

        // Do some grabbing
        cv::VideoCapture video_capture;
        if( current_file.size() > 0 )
        {
            if (!boost::filesystem::exists(current_file))
            {
                FATAL_STREAM("File does not exist");
            }

            current_file = boost::filesystem::path(current_file).generic_string();

            INFO_STREAM( "Attempting to read from file: " << current_file );
            video_capture = cv::VideoCapture( current_file );
        }
        else
        {
            //INFO_STREAM( "Attempting to capture from device: " << device );
            //video_capture = cv::VideoCapture( device );
            video_capture = cv::VideoCapture( "/home/dafu/Project/myOpenFace/1.avi" );
            // Read a first frame often empty in camera
            cv::Mat captured_image;
            video_capture >> captured_image;
        }

        if( !video_capture.isOpened() ) FATAL_STREAM( "Failed to open video source" );
        else INFO_STREAM( "Device or file opened");

        cv::Mat captured_image;
        video_capture >> captured_image;

        // If optical centers are not defined just use center of image
        if (cx_undefined)
        {
            cx = captured_image.cols / 2.0f;
            cy = captured_image.rows / 2.0f;
        }
        // Use a rough guess-timate of focal length
        if (fx_undefined)
        {
            fx = 500 * (captured_image.cols / 640.0);
            fy = 500 * (captured_image.rows / 480.0);

            fx = (fx + fy) / 2.0;
            fy = fx;
        }

        int frame_count = 0;

        // saving the videos
        cv::VideoWriter writerFace;
        if (!output_video_files.empty())
        {
            writerFace = cv::VideoWriter(output_video_files[f_n], CV_FOURCC('D', 'I', 'V', 'X'), 30, captured_image.size(), true);
        }

        // Use for timestamping if using a webcam
        int64 t_initial = cv::getTickCount();

        INFO_STREAM( "Starting tracking");
        while(!captured_image.empty())
        {
            // Reading the images
            cv::Mat_<float> depth_image;
            cv::Mat_<uchar> grayscale_image;

            if(captured_image.channels() == 3)
            {
                cv::cvtColor(captured_image, grayscale_image, CV_BGR2GRAY);
            }
            else
            {
                grayscale_image = captured_image.clone();
            }

            /// main method
            bool detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, depth_image, clnf_model, det_parameters);

            // Visualising the results
            // Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
            double detection_certainty = clnf_model.detection_certainty;

            /// detect info visualise
            visualise_tracking(captured_image, depth_image, clnf_model, det_parameters,frame_count,fx, fy, cx, cy);

            // output the tracked video
            if (!output_video_files.empty())
            {
                writerFace << captured_image;
            }

            video_capture >> captured_image;

            // detect key presses
            char character_press = cv::waitKey(1);

            // restart the tracker
            if(character_press == 'r')
            {
                clnf_model.Reset();
            }
            // quit the application
            else if(character_press=='q')
            {
                return(0);
            }

            // Update the frame count
            frame_count++;

        }

        frame_count = 0;

        // Reset the model, for the next video
        clnf_model.Reset();

        // break out of the loop if done with all the files (or using a webcam)
        if(f_n == files.size() -1 || files.empty())
        {
            done = true;
        }
    }

    return 0;
}
