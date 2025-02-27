
//OpenCV headers
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio/registry.hpp>
//MAVSDK headers
#include <mavsdk/mavsdk.h>
#include <mavsdk/plugins/telemetry/telemetry.h>
#include <mavsdk/plugins/offboard/offboard.h>
#include <mavsdk/plugins/action/action.h>
//ONNX runtime api
//#include <onnxruntime_cxx_api.h>
//standard libraries
#include <cstdint>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <vector>
#include <future>
#include <chrono>
#include <cmath>
//namespaces
using namespace cv;
using namespace cv::dnn;
using namespace std; 
using namespace mavsdk;

//-------------------------------------------------------------------
//Utility Functions
//--------------------------------------------------------------------
const string classes = "../classes.txt";
void load_classes(const string&filename, vector<string>&classlist){
	string clss;
	ifstream fp(filename);
	while(getline(fp,clss)){
		classlist.push_back(clss);
	}
}

void print_vector(vector<string>&printvec){
	for(int i = 0;i<printvec.size()-1; i++){
		cout<< printvec[i] << "\n";
	}
}
//-----------------------------------------------------------------------------------
//
//----------------------------------------------
//video/camera object detection
//----------------------------------------------
const string onnxpath = "../odm/best.onnx";
const string videopath = "../video.mp4";
const string camera = "camera";
void object_detection(const string& onnxpath, const string& option){
	//read onnx and check if there was an error
	Net yolo =  readNetFromONNX(onnxpath);
	if(yolo.empty()){
		cerr<<"Error occured when reading the ONNX file"<<endl;
	}
	//read the option for either filepath or camera, if neither then print error
	//make videocapture object given option and print error if file was unable to be opened
	VideoCapture cap;
	if(option == videopath){
		cap = VideoCapture(option);
		if(!cap.isOpened()){
			cerr<<"unable to open file"<<option<<endl;
		}
	}
	else if(option == camera){
		cap = VideoCapture(0);
		if(!cap.isOpened()){
			cerr<<"unable to open camera"<<endl;
		}
	}
	else{
		cerr <<"not a valid option"<<endl;
	}
//frame counter for modulus logic for example do something every 50 frames
//via if((current frame%50)==0){do something,or nothing};
	int currentframe = 0;
	Mat frame;
	const int targetWidth = 640;
	const int targetHeight = 640;
//vector<string> classlist = loadclasses("../classes.txt");
//detection
	for(;;){
		//capture frame
		cap >> frame;
		if(frame.empty()){
			break;
		}
		//frame to blob
		Mat blob = blobFromImage(frame, 1.0/255, Size(targetWidth,targetHeight), Scalar(0,0,0), true,false);
		//load blob
		yolo.setInput(blob);
		// forward propagation
		Mat output = yolo.forward();
		Point classIdPoint;
		double confidence;
		Mat outputRow = output.reshape(1,1);
		minMaxLoc(outputRow, 0,&confidence, 0, &classIdPoint);
		int classId = classIdPoint.x;
		string label= format("Class: %d, Conf: %.2f", classId,confidence);
		putText(frame, label, Point(20,30), FONT_HERSHEY_SIMPLEX, 1,Scalar(0,255,0),2);
		imshow("ONNX Inference", frame);
		if(waitKey(1)=='q'){
			break;
		}
		cap.release();
		destroyAllWindows();
	}
}
//----------------------------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------
// object detection but only for the purpose of feeding the model 1 image to test
// -------------------------------------------------------------------------------
void image_detector(cv::dnn::DetectionModel&model, cv::Mat frame, std::vector<int>&classids, std::vector<float>&confidences, std::vector<cv::Rect>&boxes,float  conf_threshold, float nms_threshold){
		
	model.detect(frame, classids, confidences, boxes,conf_threshold, nms_threshold);
}

void ObjectDetectionInit(const cv::dnn::DetectionModel&model, const cv::dnn:: Net&network){
	cv::dnn::DetectionModel Model = cv::dnn::DetectionModel(network);
}
//------------------------------------------------------------------------------------------------------------------
const string impth = "../images/Untitled.jpeg";

int main(int argc, char*argv[]){
	cv::Mat image = cv::imread(impth); 
	cv::dnn::Net net = readNetFromONNX(onnxpath);
	cv::dnn::DetectionModel yolo = cv::dnn::DetectionModel(net);
      	//ObjectDetectionInit(yolo,net);
	std::vector<int>classids;
	std::vector<float>confidences;
	std::vector<cv::Rect>boxes;
	float conf_thresh = 80;
	float nms_thresh = 40;
	image_detector(yolo, image,classids, confidences,boxes, conf_thresh, nms_thresh);
	
  return 0;
}
