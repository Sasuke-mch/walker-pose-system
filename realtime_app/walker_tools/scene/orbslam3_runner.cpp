// ORB-SLAM3 stereo fisheye replay adapter. GPL-3.0-or-later, matching the
// upstream ORB-SLAM3 dependency. Build against orbslam3_static_mask.patch.
// Apply that minimal dependency patch with git apply --unidiff-zero.
// Usage: orbslam3_runner manifest.json output_directory ORBvoc.txt
// The manifest contains common calibrated raw-fisheye images and masks; it
// has no Stage/ankle inputs. Nonzero mask pixels are EXCLUDED at extraction.
#include <System.h>
#include <Frame.h>
#include <MapPoint.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>

using Clock = std::chrono::steady_clock;

static double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
}

static cv::FileStorage readManifest(const std::string& path) {
    std::ifstream input(path);
    if(!input) throw std::runtime_error("cannot read manifest");
    const std::string json((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
    // OpenCV FileStorage stores integral JSON numbers as signed 32-bit ints.
    // At frame65 the valid 2166666667 ns timestamp wrapped negative. Preserve
    // the numeric value by making only these integer tokens JSON real values.
    // FileStorage still parses and validates the complete common manifest.
    const std::regex integerTimestamp(R"("timestamp_ns"(\s*:\s*)(-?[0-9]+)(\s*[,}]))");
    const std::string realTimestampJson = std::regex_replace(
        json, integerTimestamp, "\"timestamp_ns\"$1$2.0$3");
    return cv::FileStorage(realTimestampJson,
        cv::FileStorage::READ|cv::FileStorage::MEMORY|cv::FileStorage::FORMAT_JSON);
}

static cv::Mat nestedMatrix(const cv::FileNode& node, int rows, int cols) {
    if (!node.isSeq() || int(node.size()) != rows)
        throw std::runtime_error("manifest matrix shape mismatch");
    cv::Mat result(rows, cols, CV_64F);
    for (int r=0; r<rows; ++r) {
        auto row = node[r];
        if (!row.isSeq() || int(row.size()) != cols)
            throw std::runtime_error("manifest matrix row mismatch");
        for (int c=0;c<cols;++c) result.at<double>(r,c) = double(row[c]);
    }
    return result;
}

static cv::Mat vector3(const cv::FileNode& node) {
    if (!node.isSeq() || node.size()!=3)
        throw std::runtime_error("manifest translation shape mismatch");
    cv::Mat result(3,1,CV_64F);
    for (int i=0;i<3;++i) result.at<double>(i)=double(node[i]);
    return result;
}

static void writeSettings(const cv::FileNode& calibration, const std::string& path,
                          int featureCount) {
    if (std::string(calibration["model"]) != "opencv_fisheye")
        throw std::runtime_error("ORB runner requires calibrated raw OpenCV fisheye");
    const auto size=calibration["image_size"];
    const int width=int(size[0]), height=int(size[1]);
    const cv::Mat K0=nestedMatrix(calibration["left_K"],3,3);
    const cv::Mat K1=nestedMatrix(calibration["right_K"],3,3);
    const cv::Mat R=nestedMatrix(calibration["R_right_from_left"],3,3);
    const cv::Mat t=vector3(calibration["translation_right_from_left_m"]);
    // ORB's Stereo.T_c1_c2 maps right coordinates into left coordinates.
    cv::Mat leftFromRight=cv::Mat::eye(4,4,CV_64F);
    cv::Mat Rt=R.t(), tinv=-Rt*t;
    Rt.copyTo(leftFromRight(cv::Rect(0,0,3,3)));
    tinv.copyTo(leftFromRight(cv::Rect(3,0,1,3)));
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot create settings file");
    out<<std::setprecision(16)<<"%YAML:1.0\nFile.version: \"1.0\"\nCamera.type: \"KannalaBrandt8\"\n";
    for(int camera=0;camera<2;++camera) {
        const cv::Mat& K=camera==0?K0:K1;
        const auto D=calibration[camera==0?"left_D":"right_D"];
        if(D.size()!=4) throw std::runtime_error("four fisheye distortion coefficients required");
        const std::string prefix="Camera"+std::to_string(camera+1)+".";
        out<<prefix<<"fx: "<<K.at<double>(0,0)<<"\n"<<prefix<<"fy: "<<K.at<double>(1,1)<<"\n"
           <<prefix<<"cx: "<<K.at<double>(0,2)<<"\n"<<prefix<<"cy: "<<K.at<double>(1,2)<<"\n";
        for(int i=0;i<4;++i) out<<prefix<<"k"<<i+1<<": "<<double(D[i])<<"\n";
    }
    out<<"Stereo.T_c1_c2: !!opencv-matrix\n  rows: 4\n  cols: 4\n  dt: f\n  data: [";
    for(int r=0;r<4;++r) for(int c=0;c<4;++c) {
        if(r||c) out<<", ";
        out<<leftFromRight.at<double>(r,c);
    }
    out<<"]\nCamera.width: "<<width<<"\nCamera.height: "<<height<<"\n"
       <<"Camera1.overlappingBegin: 0\nCamera1.overlappingEnd: "<<width-1<<"\n"
       <<"Camera2.overlappingBegin: 0\nCamera2.overlappingEnd: "<<width-1<<"\n"
       <<"Camera.fps: 30\nCamera.RGB: 0\nStereo.ThDepth: 40.0\n"
       <<"ORBextractor.nFeatures: "<<featureCount<<"\nORBextractor.scaleFactor: 1.2\n"
       <<"ORBextractor.nLevels: 8\nORBextractor.iniThFAST: 20\nORBextractor.minThFAST: 7\n"
       <<"Viewer.KeyFrameSize: 0.05\nViewer.KeyFrameLineWidth: 1.0\nViewer.GraphLineWidth: 1.0\n"
       <<"Viewer.PointSize: 2.0\nViewer.CameraSize: 0.05\nViewer.CameraLineWidth: 2.0\n"
       <<"Viewer.ViewpointX: 0.0\nViewer.ViewpointY: -10.0\nViewer.ViewpointZ: -0.1\nViewer.ViewpointF: 2000.0\n";
}

template<class Matrix> static void jsonMatrix(std::ostream& out,const Matrix& matrix) {
    out<<"[";
    for(int r=0;r<matrix.rows();++r) {
        if(r) out<<",";
        out<<"[";
        for(int c=0;c<matrix.cols();++c) {if(c)out<<",";out<<matrix(r,c);}
        out<<"]";
    }
    out<<"]";
}

static int excludedFeatureCenters(const std::vector<cv::KeyPoint>& keys, const cv::Mat& mask) {
    int violations=0;
    for(const auto& key:keys) {
        const int x=cvRound(key.pt.x),y=cvRound(key.pt.y);
        if(x<0||y<0||x>=mask.cols||y>=mask.rows||mask.at<unsigned char>(y,x)!=0) ++violations;
    }
    return violations;
}

int main(int argc,char** argv) {
    if(argc<4 || argc>5) {
        std::cerr<<"Usage: orbslam3_runner manifest.json output_directory ORBvoc.txt [feature_count=6000]\n";
        return 2;
    }
    try {
        const std::string outDir=argv[2];
        if(mkdir(outDir.c_str(),0755)!=0 && errno!=EEXIST)
            throw std::runtime_error("cannot create output directory");
        if(std::ifstream(outDir+"/visual_rig_pose.jsonl").good())
            throw std::runtime_error("refusing to overwrite existing run");
        cv::FileStorage manifest=readManifest(argv[1]);
        if(!manifest.isOpened()) throw std::runtime_error("cannot read manifest");
        if(int(manifest["schema_version"])!=1) throw std::runtime_error("unsupported manifest version");
        const auto frames=manifest["frames"], calibration=manifest["calibration"];
        double previousTimestamp=-1.0;
        for(auto row:frames) {
            const double stamp=double(row["timestamp_ns"]);
            if(!std::isfinite(stamp)||stamp<0||stamp<=previousTimestamp)
                throw std::runtime_error("manifest timestamps must increase without integer overflow");
            previousTimestamp=stamp;
        }
        const int width=int(calibration["image_size"][0]),height=int(calibration["image_size"][1]);
        const int featureCount=argc==5?std::stoi(argv[4]):6000;
        if(featureCount<500) throw std::runtime_error("feature count below useful ORB baseline");
        writeSettings(calibration,outDir+"/orbslam3_settings.yaml",featureCount);
        cv::setNumThreads(1);
        auto initStart=Clock::now();
        // Native Settings prints both loaded camera parameter vectors before
        // tracking starts; retain sufficient digits to check formal intrinsics
        // and distinct left/right distortion against the common manifest.
        std::cout<<std::setprecision(10);
        ORB_SLAM3::System slam(argv[3],outDir+"/orbslam3_settings.yaml",ORB_SLAM3::System::STEREO,false);
        const double initializationMs=milliseconds(initStart);
        std::ofstream output(outDir+"/visual_rig_pose.jsonl");
        if(!output)throw std::runtime_error("cannot create trajectory output");
        output<<std::setprecision(10);
        const auto allStart=Clock::now();
        int tracked=0, missing=0, index=0;
        unsigned long previousMap=0; bool haveMap=false;
        for(auto row:frames) {
            const int frameId=int(row["frame_id"]),pairId=int(row["pair_id"]);
            const double timestampNs=double(row["timestamp_ns"]);
            auto start=Clock::now();
            const cv::Mat left=cv::imread(std::string(row["left_path"]),cv::IMREAD_GRAYSCALE);
            const cv::Mat right=cv::imread(std::string(row["right_path"]),cv::IMREAD_GRAYSCALE);
            const cv::Mat leftMask=cv::imread(std::string(row["left_mask_path"]),cv::IMREAD_GRAYSCALE);
            const cv::Mat rightMask=cv::imread(std::string(row["right_mask_path"]),cv::IMREAD_GRAYSCALE);
            const double inputMs=milliseconds(start);
            output<<"{\"frame_id\":"<<frameId<<",\"pair_id\":"<<pairId<<",\"timestamp_ns\":"<<std::fixed<<std::setprecision(0)<<timestampNs<<std::defaultfloat<<std::setprecision(10)
                  <<",\"backend\":\"orbslam3_stereo_fisheye\",\"world_pose_certified\":false,\"stage_used\":false,\"input_ms\":"<<inputMs;
            if(left.empty()||right.empty()||leftMask.empty()||rightMask.empty() ||
               left.size()!=cv::Size(width,height)||right.size()!=left.size()||leftMask.size()!=left.size()||rightMask.size()!=left.size()) {
                output<<",\"status\":\"input_unavailable\",\"world_from_left_camera_m\":null,\"track_ms\":null,\"map_segment_id\":null,\"observations\":[]}\n";
                output.flush();++missing;++index;continue;
            }
            auto trackStart=Clock::now();
            slam.SetStereoExclusionMasks(leftMask,rightMask);
            const auto cameraFromWorld=slam.TrackStereo(left,right,timestampNs*1e-9);
            const double trackMs=milliseconds(trackStart);
            const int state=slam.GetTrackingState();
            const unsigned long mapId=slam.GetCurrentMapId();
            const bool mapChanged=haveMap&&mapId!=previousMap;
            const bool mapUpdated=slam.MapChanged();
            previousMap=mapId;haveMap=true;
            const auto& frame=slam.GetCurrentFrameForExport();
            const int maskViolations=excludedFeatureCenters(frame.mvKeys,leftMask)+excludedFeatureCenters(frame.mvKeysRight,rightMask);
            int pointCount=0,inlierPointCount=0; bool first=true;
            std::ostringstream observations;
            observations<<std::setprecision(10);
            for(size_t i=0;i<frame.mvpMapPoints.size();++i) {
                auto point=frame.mvpMapPoints[i];
                if(!point||point->isBad())continue;
                const bool isRight=frame.Nleft>=0&&int(i)>=frame.Nleft;
                const size_t keyIndex=isRight?i-frame.Nleft:i;
                const auto& keys=isRight?frame.mvKeysRight:frame.mvKeys;
                if(keyIndex>=keys.size())continue;
                const auto xyz=point->GetWorldPos();
                if(!xyz.allFinite())continue;
                const bool outlier=i<frame.mvbOutlier.size()&&frame.mvbOutlier[i];
                if(!outlier)++inlierPointCount;
                if(!first) observations<<",";
                first=false;
                observations<<"{\"camera_index\":"<<(isRight?1:0)<<",\"u\":"<<keys[keyIndex].pt.x<<",\"v\":"<<keys[keyIndex].pt.y
                      <<",\"map_point_id\":"<<point->mnId<<",\"outlier\":"<<(outlier?"true":"false")
                      <<",\"world_xyz_m\":["<<xyz.x()<<","<<xyz.y()<<","<<xyz.z()<<"]}";
                ++pointCount;
            }
            // Native stereo initialization can announce OK with many 2D
            // features but no triangulated map point. Preserve its proposal,
            // while requiring actual finite inlier 3D support for primary use.
            const bool nativePose=state==2&&cameraFromWorld.matrix().allFinite();
            const bool valid=nativePose&&maskViolations==0&&inlierPointCount>=1;
            std::string status=maskViolations?"mask_violation":nativePose&&inlierPointCount==0?"no_3d_support":valid?"tracked":state==1?"not_initialized":state==3?"recently_lost":state==4?"lost":"unavailable";
            output<<",\"status\":\""<<status<<"\",\"tracking_state\":"<<state<<",\"track_ms\":"<<trackMs
                  <<",\"map_segment_id\":"<<mapId<<",\"map_changed\":"<<(mapChanged?"true":"false")
                  <<",\"map_update_reported\":"<<(mapUpdated?"true":"false")<<",\"pose_frame\":\"backend_map_segment\""
                  <<",\"left_keypoint_count\":"<<frame.mvKeys.size()<<",\"right_keypoint_count\":"<<frame.mvKeysRight.size()
                  <<",\"masked_feature_center_violations\":"<<maskViolations
                  <<",\"finite_inlier_map_observation_count\":"<<inlierPointCount
                  <<",\"native_pose_proposal_world_from_left_camera_m\":";
            if(nativePose)jsonMatrix(output,cameraFromWorld.inverse().matrix());else output<<"null";
            output<<",\"world_from_left_camera_m\":";
            if(valid) {jsonMatrix(output,cameraFromWorld.inverse().matrix());++tracked;} else output<<"null";
            output<<",\"observations\":["<<observations.str()<<"],\"tracked_observation_count\":"<<pointCount<<"}\n";
            output.flush();
            ++index;
            if(index%32==0)std::cout<<"processed "<<index<<" / "<<frames.size()<<", tracked "<<tracked<<std::endl;
        }
        const double replayMs=milliseconds(allStart);
        auto shutdownStart=Clock::now();slam.Shutdown();const double shutdownMs=milliseconds(shutdownStart);
        std::ofstream summary(outDir+"/runner_summary.json");
        summary<<std::setprecision(10)<<"{\"backend\":\"orbslam3_stereo_fisheye\",\"frame_count\":"<<frames.size()
               <<",\"tracked_count\":"<<tracked<<",\"input_unavailable_count\":"<<missing
               <<",\"initialization_ms\":"<<initializationMs<<",\"replay_wall_ms\":"<<replayMs
               <<",\"shutdown_ms\":"<<shutdownMs<<",\"feature_count\":"<<featureCount
               <<",\"mask_applied_at_feature_extraction\":true,\"descriptor_exclusion_halo_level_px\":22,\"minimum_finite_inlier_map_observations\":1,\"stage_used\":false,\"world_pose_certified\":false}\n";
        return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<std::endl;return 1;}
}
