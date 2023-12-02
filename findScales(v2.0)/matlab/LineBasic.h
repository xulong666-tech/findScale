#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <vector>
#include <iostream>
#include "xlTime.h"

#define Error(x)	printf("[Error]: file:%s ,line:%d ,function:%s ,message:%s \n",__FILE__,__LINE__,__FUNCSIG__,x)
typedef			std::complex<int> MPOINT;

/// @brief    	:  get line's length
/// @details  	:  
/// @param[in]	: cv::Vec4f(x1,y1,x2,y2)
/// @return   	:  int
/// @attention	:  
int
GetLineLength(cv::Vec4f line);
/// @brief    	:  get line's length
/// @details  	:  
/// @param[in]	: cv::Point p1 ,p2
/// @return   	:  
/// @attention	:  
int	
GetLineLength(cv::Point p1, cv::Point p2);
/// @brief    	:  
/// @details  	:  
/// @param[in]	: 
/// @return   	:  double
/// @attention	:  
double	
GetLineLength(cv::Point p1, cv::Point p2, bool needdoule);
/// @brief    	:  get distance between pointO and pointA .
/// @details  	:  
/// @param[in]	: 
/// @return   	:  
/// @attention	:  
float			
getDistance(cv::Point pointO, cv::Point pointA);
/// @brief    	:  get line's angle
/// @details  	:  
/// @param[in]	: 
/// @return   	:  
/// @attention	:  
double			
GetLineAngle(cv::Vec4f line);
/// @brief    	:  caculate the number of green pixels in input_img.
/// @details  	:  
/// @param[in]	: 
/// @return   	:  
/// @attention	:  
unsigned int	
get_GreenColorPointNum(cv::Mat input_img);
/// @brief    	:  assume a line is represent as 'y = kx +b' ,so this function is used to get parameter 'b'
/// @details  	:  
/// @param[in]	: 
/// @return   	:  
/// @attention	:  
double			
GetLineb(cv::Vec4f line);
/// @brief    	:  assume a line is represent as 'y = kx +b' ,so this function is used to get parameter 'b'
/// @details  	:  
/// @param[in]	: 
/// @return   	:  
/// @attention	:  
double	
GetLineb(cv::Point2f p1, cv::Point2f p2);
/// @brief    	:  assume a line is represent as 'y = kx +b' ,so this function is used to get parameter 'k'
/// @details  	:  
/// @param[in]	: 
/// @return   	:  
/// @attention	:  
double	
GetLineK(cv::Vec4f line);
/// @brief    	:  assume a line is represent as 'y = kx +b' ,so this function is used to get parameter 'k'
/// @details  	:  
/// @param[in]	: 
/// @return   	:  
/// @attention	:  
double	
GetLineK(cv::Point2f p1, cv::Point2f p2);
/// @brief    	:  get the angle betwenn line1 and line2,
/// @details  	:  
/// @param[in]	: 
/// @return   	:  angle,unit °(单位：度)
/// @attention	:  always return acute angle(锐角) of the two lines.
double	
GetLLAngle(cv::Vec4f line1, cv::Vec4f line2);
/// @brief    	:  get the cross point of the two lines.
/// @details  	:  
/// @param[in]	: 
/// @return   	:  
/// @attention	:  
cv::Point2f		
GetCrossoverPoint(double k1, double b1, double k2, double b2);
/// @brief    	:  
/// @details  	:  
/// @param[in]	: 
/// @return   	:  
/// @attention	:  
bool	
noThreeCollinear(const std::vector<cv::Vec3f> &points);
/// @brief    	:  Determine if the point is within the range of the line segment 
/// @details  	:  
/// @param[in]	: 
/// @return   	:  
/// @attention	:  
bool	
IsOnline(cv::Point2f point1, cv::Point2f point2, cv::Point2f crosspoint);
bool
IsOnline2(cv::Mat img, cv::Point PA, cv::Point PB, cv::Point Px);
/// @brief    	:  Determine if 'l1' and 'l2' is crossed in 'img';
/// @details  	:  
/// @param[in]	: 
/// @return   	:  
/// @attention	: attention the difference between  IsLineCross() and IsLineCross2().IsLineCross() uses pixels to compute while IsLineCross2() only uses geometry.
///               so IsLineCross2() is much faster than IsLineCross().
bool		
IsLineCross(cv::Mat img, cv::Vec4f l1, cv::Vec4f l2);//method1 slow
bool	
IsLineCross2(cv::Vec4f l1, cv::Vec4f l2);//method2 fast
/// @brief    	:  get the shortest distance from point to line.
/// @details  	:  
/// @param[in]	: 
/// @return   	:  
/// @attention	:  
float		
getDist_P2L(cv::Point pointP, cv::Point pointA, cv::Point pointB);
/// @brief    	:  extend line segment to endless line.
/// @details  	:  
/// @param[in]	: 
/// @return   	:  
/// @attention	:  
void		
endlessLine(cv::Mat img, cv::Point p1, cv::Point p2, const cv::Scalar& color,int thickness = 1, int lineType = cv::LINE_8, int shift = 0);
/// @brief    	:  
/// @details  	:  
/// @param[in]	: 
/// @return   	:  
/// @attention	:  
void		
CalcuRotaRectPixValue(cv::Mat img_gray, cv::RotatedRect rect, unsigned int* calcu, uchar &max_value, uchar &min_value, uchar &aver_value);
void		
myThreshold(cv::Mat img, cv::Mat &output, uchar min, uchar max);
int		
IsPointInRect(const cv::Rect& rect,const cv::Point2f& point);
bool			
IsPointInRRect(const cv::RotatedRect& RRect,const cv::Point2f& ptf);
bool	
DoesRectangleContainPoint(const cv::RotatedRect& rectangle,const cv::Point2f& point);
int		
IsRRCross(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2);//判断 点是否在 RotatedRect内
int		
countRectsNum_InViewRect(cv::Rect& view_rect,std::vector<cv::RotatedRect>& rotaRect);

std::vector<std::vector<cv::RotatedRect>::iterator>
findRects_InViewRect(cv::Rect view_rect, std::vector<cv::RotatedRect> &rotaRect);
void			
findNearestRotaRect(cv::RotatedRect father_rect, std::vector<cv::RotatedRect> &rotaRectSet, std::vector<cv::RotatedRect>::iterator &iter);
int				
drawRotatedRect(cv::Mat &img, cv::RotatedRect rect, cv::Scalar scalar, int thickness = 1, int lineType = 8, int shift = 0,int Isfull=0);
bool			
circleLeastFit(const std::vector<MPOINT> &points, double &center_x, double &center_y, double &radius);
bool		
circleLeastFit(const std::vector<cv::Point> &points, double &center_x, double &center_y, double &radius);
unsigned int
get_GreenColorPointNum(const cv::Mat& input_img);
unsigned int
get_YellowColorPointNum(const cv::Mat& input_img);


//-------------------与刻度提取有关算法函数-------------
//Draw Rotated Rects in 'img'
void 
DrawRotaRects(cv::Mat& img, std::vector<cv::RotatedRect>& box);

/*judge whether RotatedRect 'RRect1' and  'RRect1' meet condition. 
if you want to know the specific condition,please read the funtion.
*/
int 
IsRRmeetPositionCondition(cv::RotatedRect RRect1, cv::RotatedRect RRect2);
/* find the probale scales in the 'img',this function will binary the img with the
threshold from 'startThrs' to 'startThrs'
char* condition_argv[]: use to define the scales' maximun and minumn area、length/width rate
example is as follow:
	char *condition[4] = { (char*)"30",(char*)"1700",(char*)"1.5",(char*)"8" };
	findProblScale(img, box1, 20, 200, 4, condition, 1);
*/
int 
findProblScale(cv::Mat img, std::vector<cv::RotatedRect>& boxSet, int startThrs, int endThrs, int step, char* condition_argv[], bool isdebug);

std::vector<cv::RotatedRect> 
NMS(const cv::Mat& img, std::vector<cv::RotatedRect>& boxSet, int simThres, bool isdebug);
/*Pick out scales and sort them
Notice:variable 'boxSet' will be changed after calling this function.
*/
std::vector <std::vector<cv::RotatedRect>> 
PickSortscales(cv::Mat img, std::vector<cv::RotatedRect>& boxSet, bool isdebug, int sortMethod, int minChainNum, cv::Rect view_Rect = cv::Rect(0, 0, 150, 150));
/*
remove 
*/
std::vector<cv::RotatedRect> 
rmSimScale(cv::Mat img, std::vector<cv::RotatedRect> &boxSet);
typedef struct
{
	double k=0;
	double b=0;
	bool IsNone = true;
}linekb;
