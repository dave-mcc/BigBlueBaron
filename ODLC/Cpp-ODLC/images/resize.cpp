#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;


int main(){
	string img_one_path = "../Cars.jpg";
	string img_two_path = "../traffic.jpg";
	string img_three_path="../Untitled.jpeg";

	


	Mat img_one = imread(img_one_path, IMREAD_COLOR);
	Mat img_two = imread(img_two_path, IMREAD_COLOR);
	MAt img_three = imread(img_three_path, IMREAD_COLOR);
	

	string window_one_name= "";
	string window_two_name="";
	string window_three_name="";


	namedWindow(window_one_name)
	

}
