//--利用级联分类器实现简单的人脸检测
//--By Steven Meng 2012.11.29

//注意添加相应的object.dll文件，并将.xml
//文件放到当前的工作目录中

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

//--函数声明
void detectAndDisplay(Mat);

//--全局变量
string eyes_cascade_name="./haarcascade_eye_tree_eyeglasses.xml";
string face_cascade_name="./haarcascade_frontalface_alt.xml";
string window_name="Capture-Face detection";

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
RNG rng(12345);

int main()
{
	CvCapture* capture;//CvCapture为结构体
	cv::Mat frame;

	//--1.加载联机分类器文件
	if (!face_cascade.load(face_cascade_name))
	{
		printf("Error loading .xml file");
		return 1;
	}
	if (!eyes_cascade.load(eyes_cascade_name))
	{
		printf("Error loading .xml file");
		return 1;
	}

	//--2.打开摄像头
	capture=cvCaptureFromCAM(-1);
	if (capture)
	{
		printf("Open Carmer Success.");
		while (true)
		{
			frame=cvQueryFrame(capture);//获取画面
			//--3.对当前画面使用分类器进行检测
			if (!frame.empty())
			{
				detectAndDisplay(frame);
			} 
			else
			{
				printf("No Capture frame --Break!");
				break;
			}
			
			int key=waitKey(10);//按键盘下C键结束程序
			if ((char)key=='c')
			{
				break;
			}
		}
	}
	return 0;
}

//--函数定义部分
void detectAndDisplay(Mat frame)
{
	double scale = 1.3;
	std::vector<Rect> faces;//脸部
	std::vector<Rect> eyes;
	Mat frame_gray,small_image(cvRound(frame.rows/scale),cvRound(frame.cols/scale),CV_8UC1);//图片尺寸小有益于加快检测速度，提高效率
	//--转变成灰度图像、归一化
	cv::cvtColor(frame,frame_gray,CV_BGR2GRAY);
	cv::resize(frame_gray,small_image,small_image.size(),0,0,INTER_LINEAR);//将尺寸缩小到1/scale，用线性插入
	cv::equalizeHist(small_image,small_image);
	
	//-- 检测部分
	//detectMultiScale函数中small_imge表示的是要检测的输入图像为smallImg，faces表示检测到的人脸目标序列，1.1表示
	//每次图像尺寸减小的比例为1.1，2表示每一个目标至少要被检测到3次才算是真的目标(因为周围的像素和不同的窗口大
	//小都可以检测到人脸),CV_HAAR_SCALE_IMAGE表示不是缩放分类器来检测，而是缩放图像，Size(30, 30)为目标的
	//最小最大尺寸
	face_cascade.detectMultiScale(small_image,faces,1.1,2,0|CV_HAAR_SCALE_IMAGE,Size(30,30));
	
	/*--没有缩小图片尺寸的方法
	for (int i=0;i<faces.size();i++)
	{
		Point center( (faces[i].x + faces[i].width*0.5), (faces[i].y + faces[i].height*0.5) );
		ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

		Mat faceROI = frame_gray( faces[i] );
		std::vector<Rect> eyes;

		//-- 在每张人脸上检测双眼
		eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

		for( int j = 0; j < eyes.size(); j++ )
		{
			Point center( (faces[i].x + eyes[j].x + eyes[j].width*0.5), (faces[i].y + eyes[j].y + eyes[j].height*0.5 ));
			int radius = cvRound( (eyes[j].width + eyes[i].height)*0.25 );
			circle( frame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
		}
	}*/
	for (vector<Rect>::const_iterator r=faces.begin();r!=faces.end();r++)
	{
	
		Point center;
		int radius=cvRound((r->width+r->height)*0.25*scale);
		center.x=cvRound((r->x+r->width *0.5)*scale);
		center.y=cvRound((r->y+r->height *0.5)*scale);
		cv::circle(frame,center,radius,cv::Scalar(0,0,255),3,8,0);

		//检测人脸上的眼睛
		cv::Mat small_image_ROI=small_image(*r);
		eyes_cascade.detectMultiScale(small_image_ROI,eyes,1.1,2,0|CV_HAAR_SCALE_IMAGE,Size(30,30));
		for (vector<Rect>::const_iterator r1=eyes.begin();r1!=eyes.end();r1++)
		{
			int radius=cvRound((r1->width+r1->height)*0.25*scale);
			center.x=cvRound((r->x+r1->x+r1->width *0.5)*scale);
			center.y=cvRound((r->y+r1->y+r1->height *0.5)*scale);
			cv::circle(frame,center,radius,cv::Scalar(0,255,0),3,8,0);
		}	
	}
	imshow(window_name,frame);
}