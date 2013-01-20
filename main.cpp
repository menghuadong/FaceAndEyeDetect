//--���ü���������ʵ�ּ򵥵��������
//--By Steven Meng 2012.11.29

//ע�������Ӧ��object.dll�ļ�������.xml
//�ļ��ŵ���ǰ�Ĺ���Ŀ¼��

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

//--��������
void detectAndDisplay(Mat);

//--ȫ�ֱ���
string eyes_cascade_name="./haarcascade_eye_tree_eyeglasses.xml";
string face_cascade_name="./haarcascade_frontalface_alt.xml";
string window_name="Capture-Face detection";

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
RNG rng(12345);

int main()
{
	CvCapture* capture;//CvCaptureΪ�ṹ��
	cv::Mat frame;

	//--1.���������������ļ�
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

	//--2.������ͷ
	capture=cvCaptureFromCAM(-1);
	if (capture)
	{
		printf("Open Carmer Success.");
		while (true)
		{
			frame=cvQueryFrame(capture);//��ȡ����
			//--3.�Ե�ǰ����ʹ�÷��������м��
			if (!frame.empty())
			{
				detectAndDisplay(frame);
			} 
			else
			{
				printf("No Capture frame --Break!");
				break;
			}
			
			int key=waitKey(10);//��������C����������
			if ((char)key=='c')
			{
				break;
			}
		}
	}
	return 0;
}

//--�������岿��
void detectAndDisplay(Mat frame)
{
	double scale = 1.3;
	std::vector<Rect> faces;//����
	std::vector<Rect> eyes;
	Mat frame_gray,small_image(cvRound(frame.rows/scale),cvRound(frame.cols/scale),CV_8UC1);//ͼƬ�ߴ�С�����ڼӿ����ٶȣ����Ч��
	//--ת��ɻҶ�ͼ�񡢹�һ��
	cv::cvtColor(frame,frame_gray,CV_BGR2GRAY);
	cv::resize(frame_gray,small_image,small_image.size(),0,0,INTER_LINEAR);//���ߴ���С��1/scale�������Բ���
	cv::equalizeHist(small_image,small_image);
	
	//-- ��ⲿ��
	//detectMultiScale������small_imge��ʾ����Ҫ��������ͼ��ΪsmallImg��faces��ʾ��⵽������Ŀ�����У�1.1��ʾ
	//ÿ��ͼ��ߴ��С�ı���Ϊ1.1��2��ʾÿһ��Ŀ������Ҫ����⵽3�β��������Ŀ��(��Ϊ��Χ�����غͲ�ͬ�Ĵ��ڴ�
	//С�����Լ�⵽����),CV_HAAR_SCALE_IMAGE��ʾ�������ŷ���������⣬��������ͼ��Size(30, 30)ΪĿ���
	//��С���ߴ�
	face_cascade.detectMultiScale(small_image,faces,1.1,2,0|CV_HAAR_SCALE_IMAGE,Size(30,30));
	
	/*--û����СͼƬ�ߴ�ķ���
	for (int i=0;i<faces.size();i++)
	{
		Point center( (faces[i].x + faces[i].width*0.5), (faces[i].y + faces[i].height*0.5) );
		ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

		Mat faceROI = frame_gray( faces[i] );
		std::vector<Rect> eyes;

		//-- ��ÿ�������ϼ��˫��
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

		//��������ϵ��۾�
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