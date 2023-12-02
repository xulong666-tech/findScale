// findScales.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include "LineBasic.h"
#include <iostream>
#include <chrono>//use this library to  calculate time consume of algorithm
#include "xlTime.h"
#include <thread>
#include "xlInstrumentation.h"
using namespace std;
using namespace cv;
using namespace xl;
#define DEBUG 1

void ThreadVideo()
{
	VideoCapture capture;
	Mat frame;
	capture.open(0);

	while (capture.read(frame))
	{
		imshow("output", frame);
		waitKey(10);
	}

}
std::string ImgTestPath = "Img/ImgTest/";
std::string ImgResultPath = "Img/ImgResult/";
std::vector<std::string> ImgSet = { 
	"tp8.png", 
	"221.bmp",
	"19.bmp",
	"2.bmp",
	"3.jpeg",
	"bd1.png",
	"bd2.png",
	"bd3.png",
	"bd4.png",
	"bd5.png",
	"bd6.png",
	"bd7.png",
	"bd8.png",
	"bd9.png",
	"bd10.png",
	"bd11.png",
	"bd12.png",
	"bd13.png",
	"bd14.png",
	"bd15.png",
	"bd16.png",
	"handwriteScales.png",
	"tb_2.jpg",
	"tb_7.png",
	"tb_8.png",
};
int main()
{
	for (std::string ImgName : ImgSet)
	{
		if (ImgName == "3.jpeg")
		{
			//std::string ImgName = ImgSet[2];
			Mat img = imread(ImgTestPath + ImgName);
			Mat img_show = img.clone();

			//1.1 define variable
			vector<RotatedRect> probboxs;
			vector<RotatedRect> aftNMS;
			vector <vector<RotatedRect>> ScalesChain;

			//1.2 set scales' conditions : minimun area size ,maximun area size, minum length:width rates , maximun length:width rates
			char *condition[4] = { (char*)"15",(char*)"1700",(char*)"1.5",(char*)"16" };

			//1.3 find probable scales,scales stored in probboxs, '20','200','4' represent binary the img with threshold from 20 to 200 with step 4. 
			CoutTime(findProblScale(img, probboxs, 20, 200, 3, condition, DEBUG));//find probably scalars,and save as RotatedRect vector.
	
			//1.4 use NMS algorithm to remove crossed scales,and remain only one scale of the crossed scales(with high confidence level).
			// '3' represent remove 
			CoutTime(aftNMS = NMS(img, probboxs,7, DEBUG));//remov redundency box,promise each box is not crossed.
	
			//1.5 pick out all the real scales,and sort them.
			CoutTime(ScalesChain = PickSortscales(img, aftNMS, DEBUG, 0, 25));
	

			get_YellowColorPointNum(img);
			//1.6 draw the result of the algorithm.
			if (!ScalesChain.empty())
			{
				for (int i=0;i<ScalesChain.size();i++)
				{
			
					DrawRotaRects(img_show, ScalesChain[i]);
					imshow("img_show"+to_string(i), img_show);
					long long rowNum = 0;
					Mat MaskImg = getRotaRectsMaskImg(img_show, ScalesChain[i], Scalar(0, 0, 255));
					//拼接图片
					Mat resultImg(img.rows + img_show.rows + MaskImg.rows, img_show.cols, img_show.type());
					img.rowRange(0, img.rows).copyTo(resultImg.rowRange(rowNum, img.rows)); rowNum += img.rows;
					img_show.rowRange(0, img_show.rows).copyTo(resultImg.rowRange(rowNum, rowNum + img_show.rows)); rowNum += img_show.rows;
					MaskImg.rowRange(0, MaskImg.rows).copyTo(resultImg.rowRange(rowNum, resultImg.rows));
					imwrite(ImgResultPath + "Result" + ImgName , resultImg);
				}
		
		
			}
			waitKey(0);
		}
	}
}

