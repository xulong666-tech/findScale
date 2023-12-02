#include "LineBasic.h"
using namespace std;
using namespace cv;
//计算一条直线的长度的平方，用于比较长度
int GetLineLength(Vec4f line)
{
	int dis = (line[0] - line[2])*(line[0] - line[2]) + (line[1] - line[3])*(line[1] - line[3]);
	return dis;
}
int GetLineLength(Point p1, Point p2)
{
	int dis = (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y);
	return dis;
}
double GetLineLength(Point p1, Point p2, bool needdoule)
{
	double dis = (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y);
	return sqrt(dis);
}
//两点间距离公式
float getDistance(Point pointO, Point pointA)
{
	float distance;
	distance = powf((pointO.x - pointA.x), 2) + powf((pointO.y - pointA.y), 2);
	distance = sqrtf(distance);
	//屏幕分辨率为1366*768，一个像素约为0.214mm，所以这里乘以0.214转化为实际尺寸，当然分辨率不同，一个像素表示的实际长度也不同
	return distance;
}

//计算一条直线的角度（arctank）
double GetLineAngle(Vec4f line)
{
	double angle = atan2((line[0] - line[2]), (line[1] - line[3]))* 180.0 / 3.1416;
	return angle;
}

double GetLineK(Vec4f line)
{
	return (line[3] - line[1]) / (line[0] - line[2]);
}

double GetLineK(Point2f p1, Point2f p2)
{
	return (p2.y - p1.y) / (p1.x - p2.x);
}

double GetLineb(Vec4f line)
{
	return -line[1] - (-line[3] + line[1]) / (line[2] - line[0])*line[0];
}

double GetLineb(Point2f p1, Point2f p2)
{
	return -p1.y - (-p2.y + p1.y) / (p2.x - p1.x)*p1.x;
}
double GetLLAngle(Vec4f line1, Vec4f line2)
{
	//线段line1 对应的向量a
	double Va_x = line1[0] - line1[2];
	double Va_y = line1[1] - line1[3];
	//
	//线段line2 对应的向量b
	double Vb_x = line2[0] - line2[2];
	double Vb_y = line2[1] - line2[3];
	//printf("Va=(%f,%f),Vb=(%f,%f)", Va_x, Va_y, Vb_x, Vb_y);
	//求向量a ,b 点积
	double dotP = Va_x * Vb_x + Va_y * Vb_y;
	//printf("dotP=%f", dotP);
	//向量a,b的模
	double Ma = sqrt(Va_x*Va_x+ Va_y* Va_y);
	double Mb = sqrt(Vb_x*Vb_x+ Vb_y* Vb_y);
	//printf("Ma=%f,Mb=%f", Ma, Mb);
	//求夹角
	if (Ma*Mb != 0)
	{
		double R = acos(abs(dotP / (Ma*Mb)));//单位为弧度
		double agl = (180.0/3.1415926)*R;//转化为度°
		//printf("agl=%f", agl);
		return agl;
	}
	else
	{
		return 0.0; 
	}

}
//求两条直线的交点
Point2f GetCrossoverPoint(double k1, double b1, double k2, double b2)
{
	Point2f out;
	out.x = (b1 - b2) / (k2 - k1);
	out.y = out.x *k1 + b1;
	out.y = -out.y;//注意opencv坐标系定义规则
	return out;
}

//求点是不是在线段的范围内
bool IsOnline(Point2f point1, Point2f point2,Point2f crosspoint)
{
	double xmax = max(point1.x, point2.x);
	double xmin = min(point1.x, point2.x);
	double ymax = max(point1.y, point2.y);
	double ymin = min(point1.y, point2.y);
	double x = crosspoint.x;
	double y = crosspoint.y;
	if (x <= xmax && x >= xmin)
	{
		if (y <= ymax && y >= ymin)
			return true;
	}
	return false;
}
// 判断两条线是否相交
bool IsLineCross(Mat img,Vec4f l1, Vec4f l2)
{

	/*line(img, Point(l1[0], l1[1]), Point(l1[2], l1[3]),Scalar(255,0,0),2);
	line(img, Point(l2[0], l2[1]), Point(l2[2], l2[3]), Scalar(255, 0, 0), 2);
	imshow("linecross",img);
	waitKey(1);*/
	LineIterator lit1(img,Point(l1[0], l1[1]), Point(l1[2], l1[3]));
	
	for (int i = 0; i < lit1.count; i++, lit1++)
	{
		LineIterator lit2(img, Point(l2[0], l2[1]), Point(l2[2], l2[3]));
		for (int j = 0; j < lit2.count; j++, lit2++)
		{
			if (lit1.pos() == lit2.pos())
			{
				return true;
			}
		}


	}
	return false;

}
bool IsLineCross2(Vec4f l1, Vec4f l2)
{
	//快速排斥实验
	if ((l1[0] > l1[2] ? l1[0] : l1[2]) < (l2[0] < l2[2] ? l2[0] : l2[2]) ||
		(l1[1] > l1[3] ? l1[1] : l1[3]) < (l2[1] < l2[3] ? l2[1] : l2[3]) ||
		(l2[0] > l2[2] ? l2[0] : l2[2]) < (l1[0] < l1[2] ? l1[0] : l1[2]) ||
		(l2[1] > l2[3] ? l2[1] : l2[3]) < (l1[1] < l1[3] ? l1[1] : l1[3]))
	{
		return false;
	}
	//跨立实验
	if ((((l1[0] - l2[0])*(l2[3] - l2[1]) - (l1[1] - l2[1])*(l2[2] - l2[0]))*
		((l1[2] - l2[0])*(l2[3] - l2[1]) - (l1[3] - l2[1])*(l2[2] - l2[0]))) > 0 ||
		(((l2[0] - l1[0])*(l1[3] - l1[1]) - (l2[1] - l1[1])*(l1[2] - l1[0]))*
		((l2[2] - l1[0])*(l1[3] - l1[1]) - (l2[3] - l1[1])*(l1[2] - l1[0]))) > 0)
	{
		return false;
	}
	return true;
}

/***** 点到直线的距离:P到AB的距离*****/
//P为线外一点，AB为线段两个端点
float getDist_P2L(Point pointP, Point pointA, Point pointB)
{
	//求直线方程
	int A = 0, B = 0, C = 0;
	A = pointA.y - pointB.y;
	B = pointB.x - pointA.x;
	C = pointA.x*pointB.y - pointA.y*pointB.x;
	//代入点到直线距离公式
	float distance = 0;
	distance = ((float)abs(A*pointP.x + B * pointP.y + C)) / ((float)sqrtf(A*A + B * B));
	return distance;
}
bool IsOnline2(Mat img, Point PA, Point PB, Point Px)
{
	LineIterator lit(img, PA, PB);
	for (int i = 0; i < lit.count; ++i,++lit)
	{
		if (lit.pos().x == Px.x & lit.pos().y == Px.y)
		{
			return true;
		}
	}
	return false;
}
void endlessLine(Mat img, Point p1, Point p2, const Scalar& color,int thickness, int lineType, int shift)
{

	Point p3(0, 0);
	int img_width = img.cols;
	int img_height = img.rows;
	circle(img, p1, 2, Scalar(255), 2);
	circle(img, p2, 2, Scalar(255), 2);

	//上边
	for (int i = 0; i < img_width; i++)
	{
		p3.x = i;
		p3.y = 0;
		if (IsOnline2(img, p1, p3, p2))
		{
			circle(img, p3, 2, Scalar(255), 2);
			line(img, p1, p3,color,thickness,lineType,shift);
			break;
		}
		if (IsOnline2(img, p2, p3, p1))
		{
			circle(img, p3, 2, Scalar(255), 2);
			line(img, p2, p3, color, thickness, lineType, shift);
			break;
		}
	}
	//左边
	for (int i = 0; i < img_height; i++)
	{
		p3.x = 0;
		p3.y = i;
		if (IsOnline2(img, p1, p3, p2))
		{
			circle(img, p3, 2, Scalar(255), 2);
			line(img, p1, p3, color, thickness, lineType, shift);
			break;
		}
		if (IsOnline2(img, p2, p3, p1))
		{
			circle(img, p3, 2, Scalar(255), 2);
			line(img, p2, p3, color, thickness, lineType, shift);
			break;
		}
	}
	//下边
	for (int i = 0; i < img_width; i++)
	{
		p3.x = i;
		p3.y = img_height;
		if (IsOnline2(img, p1, p3, p2))
		{
			circle(img, p3, 2, Scalar(255), 2);
			line(img, p1, p3, color, thickness, lineType, shift);
			break;
		}
		if (IsOnline2(img, p2, p3, p1))
		{
			circle(img, p3, 2, Scalar(255), 2);
			line(img, p2, p3, color, thickness, lineType, shift);
			break;
		}
	}
	//右边
	for (int i = 0; i < img_height; i++)
	{
		p3.x = img_width;
		p3.y = i;
		if (IsOnline2(img, p1, p3, p2))
		{
			circle(img, p3, 2, Scalar(255), 2);
			line(img, p1, p3, color, thickness, lineType, shift);
			break;
		}
		if (IsOnline2(img, p2, p3, p1))
		{
			circle(img, p3, 2, Scalar(255), 2);
			line(img, p2, p3, color, thickness, lineType, shift);
			break;
		}
	}
}
//Note:calcu must be inited as 0
void CalcuRotaRectPixValue(Mat img_gray, RotatedRect rect, unsigned int* calcu, uchar &max_value, uchar &min_value,uchar &aver_value)
{
	//如果不是灰度图则转化为灰度图
	if (img_gray.channels() != 1)
	{
		cvtColor(img_gray, img_gray, CV_RGB2GRAY);
	}

	cv::Point2f* vertices = new cv::Point2f[4];
	rect.points(vertices);
	//for (int j = 0; j < 4; j++) {  //绘制最小外接轮廓的四条边

	//	line(img_gray, Point(vertices[j]), Point(vertices[(j + 1) % 4]), Scalar(255, 0, 255), 2);

	//}
	//取得对边像素点
	LineIterator lit1(img_gray, vertices[0], vertices[1]);
	LineIterator lit2(img_gray, vertices[3], vertices[2]);
	unsigned int max = 0, min = 255;
	long totalPixNum=0;
	max_value = 0; min_value = 255;
	long sum = 0;
	for (int i = 0; i < lit1.count; ++i, ++lit1, ++lit2)
	{
		LineIterator litx(img_gray, lit1.pos(), lit2.pos());
		for (int i = 0; i < litx.count; ++i, ++litx)
		{
			//printf("litx.pox().x=%d,litx.pox().y=%d", litx.pos().x, litx.pos().y);
			uchar num =  img_gray.at<uchar>(litx.pos().y,litx.pos().x);
			sum += num;
			calcu[num]+=1;
			totalPixNum++;
			if (calcu[num] > max) max = calcu[num];
			if (calcu[num] < min) min = calcu[num];
			if (num > max_value) max_value = num;
			if (num < min_value) min_value = num;
			//检验是否溢出
		}
		//line(img_gray, lit1.pos(), lit2.pos(), Scalar(100), 2);
	}
	if (totalPixNum != 0)
	{
		aver_value = sum / totalPixNum;
	}
	
	//printf("max=%d",max);
	uchar pad_bottom = 40;
	uchar pad_up = 40;
	uchar pad_left = 40;
	uchar pad_right = 80;
	int axis_x = 1000;
	int axis_y = 500;
	int img_width= axis_x+ pad_left+ pad_right;
	int img_height= axis_y+ pad_up+ pad_bottom;
	Mat curve_img(Size(img_width, img_height), CV_8UC3,Scalar(0,0,0));
	putText(curve_img, "0", Point(pad_left, curve_img.rows-20), 0, 1, Scalar(0, 0, 255), 2);
	putText(curve_img, "255", Point(curve_img.cols- pad_right, curve_img.rows - 20), 0, 1, Scalar(0, 0, 255), 2);
	putText(curve_img, "total pixs="+to_string(totalPixNum), Point(40, 40), 0, 1, Scalar(0, 0, 255), 2);
	for (int i = 0; i < 256-1; i++)
	{
		
		float rate_x = axis_x/256.0 ;
		float rate_y = axis_y/(float)max;
		Point p1(i * rate_x + pad_left, curve_img.rows - calcu[i] * rate_y - pad_bottom);
		int j = i + 1;
		Point p2(j * rate_x + pad_left, curve_img.rows - calcu[j] * rate_y - pad_bottom);
		line(curve_img,p1,p2,Scalar(0,255,0),2);
		/*curve_img.at<Vec3b>(curve_img.rows-calcu[i]- pad_bottom,i*rate_x + pad_left)[0] = 0;
		curve_img.at<Vec3b>(curve_img.rows - calcu[i]- pad_bottom, i*rate_x + pad_left)[1] = 255;
		curve_img.at<Vec3b>(curve_img.rows - calcu[i]- pad_bottom, i*rate_x + pad_left)[2] = 0;*/
		//printf("...%d....", calcu[i]);
	}
	imshow("curve_img", curve_img);
	waitKey(1);
}
//对设定的阈值范围内的像素置255，其他归零
void myThreshold(Mat img,Mat &output,uchar min,uchar max)
{
	//如果不是灰度图则转化为灰度图
	output = img.clone();
	if (img.channels() != 1)
	{
		cvtColor(output, output, CV_RGB2GRAY);
	}
	for (int i = 0; i < output.rows; i++)
	{
		for (int j = 0; j < output.cols; j++)
		{
			uchar pix_value = output.at<uchar>(i, j);
			//if (pix_value>= min+70 & pix_value <= max + 50)
			if (pix_value >= min & pix_value <= max)
			{
				output.at<uchar>(i, j)=255;
			}
			else
			{
				output.at<uchar>(i, j) = 0;
			}
		}
	}
}
int IsPointInRect(const Rect& rect,const Point2f& point)
{


	//判断某点与矩形的位置关系
	if (point.x > rect.tl().x&&point.x<rect.br().x&&point.y>rect.tl().y&&point.y < rect.br().y)
	{
		//std::cout << "点在矩形内" << std::endl;
		return 0;
	}
	else if ((point.x == rect.tl().x || point.x == rect.br().x) && (point.y >= rect.tl().y&&point.y <= rect.br().y))
	{
		//std::cout << "点在矩形边界上" << std::endl;
		return 1;
	}
	else if ((point.y == rect.tl().y || point.y == rect.br().y) && (point.x >= rect.tl().x&&point.x <= rect.br().x))
	{
		//std::cout << "点在矩形边界上" << std::endl;
		return 1;
	}
	else
	{
		//std::cout << "点在矩形外" << std::endl;
		return 2;

	}
}
bool IsPointInRRect(const RotatedRect& RRect,const Point2f& ptf)
{
	Point2f ptf4[4];
	float fAngle = RRect.angle;
	RRect.points(ptf4);
	Point2f ptf4Vector[4];
	int nQuadrant[4] = { 0 };
	fAngle *= CV_PI / 180.0*(-1);

	for (int idx = 0; idx < 4; idx++)
	{
		float fDifx = float(ptf.x - ptf4[idx].x);
		float fDify = float(ptf.y - ptf4[idx].y);
		int nDifx = fDifx * cos(fAngle) - fDify * sin(fAngle);
		int nDify = fDifx * sin(fAngle) + fDify * cos(fAngle);

		//第一象限
		if (nDifx >= 0 && nDify >= 0)
			nQuadrant[0]++;
		//第二象限
		if (nDifx < 0 && nDify >= 0)
			nQuadrant[1]++;
		//第三象限
		if (nDifx < 0 && nDify < 0)
			nQuadrant[2]++;
		//第四象限
		if (nDifx > 0 && nDify < 0)
			nQuadrant[3]++;
	}
	//判断四个向量是否在相邻的两个象限
	int firstIdx = -1;
	int secIdx = -1;
	int countNum = 0;
	for (int idx = 0; idx < 4; idx++)
	{
		if (nQuadrant[idx] != 0)
		{
			if (firstIdx == -1)
				firstIdx = idx;
			else if (secIdx == -1 && firstIdx != -1)
				secIdx = idx;

			countNum++;
		}
	}

	if (countNum <= 2)
		if (abs(firstIdx - secIdx) == 1 || abs(firstIdx - secIdx) == 3 || (countNum == 1 && (firstIdx == -1 || secIdx == -1)))
			return false;
	return true;
}
bool DoesRectangleContainPoint(const RotatedRect& rectangle,const Point2f& point) {

	//Get the corner points.
	Point2f corners[4];
	rectangle.points(corners);

	//Convert the point array to a vector.
	//https://stackoverflow.com/a/8777619/1997617
	Point2f * lastItemPointer = (corners + sizeof corners / sizeof corners[0]);
	vector <Point2f > contour(corners, lastItemPointer);

	//Check if the point is within the rectangle.
	double indicator = pointPolygonTest(contour, point, false);
	bool rectangleContainsPoint = (indicator >= 0);
	return rectangleContainsPoint;
}

int countRectsNum_InViewRect(Rect& view_rect,vector<RotatedRect>& rotaRect)
{
	vector<RotatedRect>::iterator it;
	int count = 0;
	for (it = rotaRect.end(); it != rotaRect.begin(); it--)
	{
		RotatedRect rectx = *it;
		if (IsPointInRect(view_rect, rectx.center) <= 1)
		{
			count++;
		}
	}
	return count;
}
vector<vector<RotatedRect>::iterator> findRects_InViewRect(Rect view_rect, vector<RotatedRect> &rotaRect)
{
	vector<vector<RotatedRect>::iterator> Viter;
	vector<RotatedRect>::iterator it;
	for (it = rotaRect.begin(); it != rotaRect.end(); it++)
	{
		RotatedRect rectx = *it;
		if (IsPointInRect(view_rect, rectx.center) <= 1)
		{
			Viter.push_back(it);
		}
	}
	return Viter;
}

void findNearestRotaRect(RotatedRect father_rect, vector<RotatedRect> &rotaRectSet, vector<RotatedRect>::iterator &iter)
{
	int minDist=10000;
	int Id = 0;
	vector<RotatedRect>::iterator iter_tmp;
	for (iter = rotaRectSet.begin(); iter != rotaRectSet.end(); iter++)
	{
		RotatedRect rotaRectx = *iter;
		int distmp = getDistance(father_rect.center, rotaRectx.center);
		if (distmp < minDist)
		{
			minDist = distmp;
			iter_tmp = iter;
		}
	}
	iter = iter_tmp;
}

int drawRotatedRect(Mat &img, RotatedRect rect, Scalar scalar, int thickness, int lineType, int shift,int Isfull)
{
	//画外框
	Point2f ps[4];//外接矩形四个端点的集合
	rect.points(ps);  //将最小外接矩形的四个端点复制给ps数组
	for (int j = 0; j < 4; j++)
	{
		line(img, Point(ps[j]), Point(ps[(j + 1) % 4]), scalar, thickness, lineType, shift);
	}
	//如果要填充则进行填充
	if (Isfull)
	{
		
		LineIterator lit1(img, ps[0], ps[1]);
		LineIterator lit2(img, ps[3], ps[2]);
		for (int i = 0; i < lit1.count; ++i, ++lit1, ++lit2)
		{
			line(img, lit1.pos(), lit2.pos(), scalar, thickness, lineType, shift);
		}
	}
	return 0;
}
int IsRRCross(const RotatedRect& rect1,const RotatedRect& rect2)
{
	Point2f ps1[4], ps2[4];//外接矩形四个端点的集合
	rect1.points(ps1);  //将最小外接矩形的四个端点复制给ps数组
	rect2.points(ps2);
	for (int i = 0; i < 4; i++)
	{
		Vec4f l1(ps1[i].x, ps1[i].y, ps1[(i + 1) % 4].x, ps1[(i + 1) % 4].y);
		for (int j = 0; j < 4; j++)
		{
			Vec4f l2(ps2[j].x, ps2[j].y, ps2[(j + 1) % 4].x, ps2[(j + 1) % 4].y);

			if (IsLineCross2(l1, l2))
			{
				return 1;
			}
		}
	}

	if (IsPointInRRect(rect1, rect2.center) | IsPointInRRect(rect2, rect1.center))
	{
		return 1;
	}
	else
	{
		return 0;
	}

}
bool circleLeastFit(const std::vector<MPOINT> &points, double &center_x, double &center_y, double &radius)
{
	center_x = 0.0f;
	center_y = 0.0f;
	radius = 0.0f;
	if (points.size() < 3)
	{
		return false;
	}

	double sum_x = 0.0f, sum_y = 0.0f;
	double sum_x2 = 0.0f, sum_y2 = 0.0f;
	double sum_x3 = 0.0f, sum_y3 = 0.0f;
	double sum_xy = 0.0f, sum_x1y2 = 0.0f, sum_x2y1 = 0.0f;

	int N = points.size();
	for (int i = 0; i < N; i++)
	{
		double x = points[i].real();
		double y = points[i].imag();
		double x2 = x * x;
		double y2 = y * y;
		sum_x += x;
		sum_y += y;
		sum_x2 += x2;
		sum_y2 += y2;
		sum_x3 += x2 * x;
		sum_y3 += y2 * y;
		sum_xy += x * y;
		sum_x1y2 += x * y2;
		sum_x2y1 += x2 * y;
	}

	double C, D, E, G, H;
	double a, b, c;

	C = N * sum_x2 - sum_x * sum_x;
	D = N * sum_xy - sum_x * sum_y;
	E = N * sum_x3 + N * sum_x1y2 - (sum_x2 + sum_y2) * sum_x;
	G = N * sum_y2 - sum_y * sum_y;
	H = N * sum_x2y1 + N * sum_y3 - (sum_x2 + sum_y2) * sum_y;
	a = (H * D - E * G) / (C * G - D * D);
	b = (H * C - E * D) / (D * D - G * C);
	c = -(a * sum_x + b * sum_y + sum_x2 + sum_y2) / N;

	center_x = a / (-2);
	center_y = b / (-2);
	radius = sqrt(a * a + b * b - 4 * c) / 2;
	return true;
}
bool circleLeastFit(const std::vector<Point> &points, double &center_x, double &center_y, double &radius)
{
	vector<MPOINT> pointS;
	for (int i = 0; i < points.size(); i++)
	{
		MPOINT point_tmp(points[i].x, points[i].y);
		pointS.push_back(point_tmp);
	}
	return circleLeastFit(pointS, center_x, center_y,radius);
}
unsigned int get_GreenColorPointNum(const Mat& input_img)
{
	Mat img_hsv;
	//绿色
	Scalar scalarL = Scalar(35, 43, 46);
	Scalar scalarH = Scalar(77, 255, 255);

	unsigned int cnt = 0;
	cvtColor(input_img, img_hsv, COLOR_BGR2HSV); // 将BGR图像转换为HSV格式
	Mat mask;
	inRange(img_hsv, scalarL, scalarH, mask);
	//imshow("test", img);
	//单通道图像,at(y , x)索引是先行（y轴） ， 后列（x轴）
	for (int h = 0; h < mask.rows; ++h)
	{
		for (int w = 0; w < mask.cols / 2; ++w)
		{
			//mask.at<uchar>(h, w) = 128;
			if (mask.at<uchar>(h, w) >= 200)
			{
				cnt++;
			}

		}
	}
	if (cnt > 3000)
	{
		printf("灯光检测合格");
	}
	//printf("i=%d", cnt);
	//imshow("hsv", mask);
	//waitKey(0);
	return cnt;
}
unsigned int get_YellowColorPointNum(const Mat& input_img)
{
	Mat img_hsv;
	//黄色
	Scalar scalarL = Scalar(26, 43, 46);
	Scalar scalarH = Scalar(50, 255, 255);

	unsigned int cnt = 0;
	cvtColor(input_img, img_hsv, COLOR_BGR2HSV); // 将BGR图像转换为HSV格式
	Mat mask;
	inRange(img_hsv, scalarL, scalarH, mask);
	//imshow("test", img_hsv);
	//单通道图像,at(y , x)索引是先行（y轴） ， 后列（x轴）
	for (int h = 0; h < mask.rows; ++h)
	{
		for (int w = 0; w < mask.cols / 2; ++w)
		{
			//mask.at<uchar>(h, w) = 128;
			if (mask.at<uchar>(h, w) >= 200)
			{
				cnt++;
			}

		}
	}
	if (cnt > 3000)
	{
		printf("灯光检测合格");
	}
	printf("i=%d", cnt);
	//imshow("hsv", mask);
	//waitKey(1);
	return cnt;
}

void DrawRotaRects(Mat &img, vector<RotatedRect> &box)
{
	Point2f ps2[4];//外接矩形四个端点的集合
	if (box.empty())
	{
		Error("box chains is empty!");
		return;
	}
	if (box[box.size() - 1].center.x > box[0].center.x)
	{
		//printf("---1\n");
		for (int i = 0; i < box.size(); i++)
		{
			box[i].points(ps2);  //将最小外接矩形的四个端点复制给ps数组
			for (int j2 = 0; j2 < 4; j2++)
			{
				line(img, Point(ps2[j2]), Point(ps2[(j2 + 1) % 4]), Scalar(255, 0, 255), 2);
				//cv::putText(show,to_string(j), ps[j], 0, 1,Scalar(0, 0, 255), 2);  //将点集按顺序显示在图上
			}
			cv::putText(img, to_string(i), ps2[0], 0, 1, Scalar(0, 0, 255), 2);
		}
	}

	else
	{
		//printf("---2\n");
		for (int i = 0; i < box.size(); i++)
		{
			box[i].points(ps2);  //将最小外接矩形的四个端点复制给ps数组
			for (int j2 = 0; j2 < 4; j2++)
			{
				line(img, Point(ps2[j2]), Point(ps2[(j2 + 1) % 4]), Scalar(255, 0, 255), 2);
				//cv::putText(show,to_string(j), ps[j], 0, 1,Scalar(0, 0, 255), 2);  //将点集按顺序显示在图上
			}
			cv::putText(img, to_string(box.size() - 1 - i), ps2[0], 0, 1, Scalar(0, 0, 255), 2);
		}
	}
}
//判断两个 RotatedRect 是否满足 位置关系
int IsRRmeetPositionCondition(RotatedRect RRect1, RotatedRect RRect2)
{
	// get length line of rectx 
	Point2f ps[4];//外接矩形四个端点的集合
	RRect1.points(ps);  //将最小外接矩形的四个端点复制给ps数组
	int cnt2Choose = RRect1.size.width < RRect1.size.height ? 1 : 3;//确定长边的另一个端点
	Vec4f lx = { ps[0].x,ps[0].y,ps[cnt2Choose].x,ps[cnt2Choose].y };

	RRect2.points(ps);  //将最小外接矩形的四个端点复制给ps数组
	cnt2Choose = RRect2.size.width < RRect2.size.height ? 1 : 3;//确定长边的另一个端点
	Vec4f ly = { ps[0].x,ps[0].y,ps[cnt2Choose].x,ps[cnt2Choose].y };
	Vec4f lz = { RRect1.center.x,RRect1.center.y,RRect2.center.x,RRect2.center.y };
	if (GetLLAngle(lx, ly) < 20 & \
		GetLLAngle(lx, lz) > 45)
		/*if (GetLLAngle(lx, ly) < 35 & \
			GetLLAngle(lx, lz) > 45)*/
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

//char* condition_argv[] 用来存储参数条件
//condition_argv[0] 为boxarea 的下限值，condition_argv[1] 为boxarea 的上限值
//condition_argv[2] 为长宽比(rate_lw) 的下限值，condition_argv[3] 为长宽比(rate_lw) 的上限值
int findProblScale(Mat img, vector<RotatedRect>& boxSet, int startThrs, int endThrs, int step, char* condition_argv[], bool isdebug)
{
	Mat grayimg_ori, thre_img, show = img.clone();
	int areasize_min, areasize_max, rate_lw_min, rate_lw_max;
	//1.1 check input arguments: startThrs endThrs step
	if (startThrs > 255 | startThrs < 0 | \
		endThrs>255 | endThrs < 0 | \
		step < 0 | step>255 | startThrs>endThrs)
	{
		Error("input arguments error!");
		return -1;
	}
	//1.2 check image
	if (img.channels() == 3){
		cvtColor(img, grayimg_ori, CV_RGB2GRAY);//转换为灰度图
	}
	else if (img.channels() == 1) { 
		grayimg_ori = img;
	}//do nothing
	else{
		Error("image error");
		return -1;
	}
	//1.3 check input arguments:condition_argv
	if (condition_argv == NULL){
		//printf("[Error]:" "\__LINE__\" __FUNCSIG__);
		
		return -1;
	}
	else{
		areasize_min = atoi(condition_argv[0]);//get parameter areasize_min
		areasize_max = atoi(condition_argv[1]);//get parameter areasize_max
		rate_lw_min = atoi(condition_argv[2]);//get parameter rate_lw_min
		rate_lw_max = atoi(condition_argv[3]);//get parameter rate_lw_max
	}
	//反色处理
	//int height = img.rows;
	//int width = img.cols;
	//for (int i = 0; i < height; i++)
	//{
	//	for (int j = 0; j < width; j++)
	//	{
	//		grayimg_ori.at<uchar>(i, j) = 255 - grayimg_ori.at<uchar>(i, j);   //对每一个像素反转
	//	}
	//}
	//2.1  filter
	medianBlur(grayimg_ori, grayimg_ori, 3);//median filter
	//2.2 find probable scales
	for (int thrsValu = startThrs; thrsValu <= endThrs; thrsValu += step)
	{

		vector<vector<Point>> contours;    //储存轮廓
		vector<Vec4i> hierarchy;
		//2.2.1 binary the image.
		threshold(grayimg_ori, thre_img, thrsValu, 255, ADAPTIVE_THRESH_MEAN_C);
		if (isdebug)
		{
			namedWindow("findthresimg", WINDOW_NORMAL);
			imshow("findthresimg", thre_img);
			waitKey(1);
		}
		//2.2.2 find the contours
		findContours(thre_img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);    //获取轮廓
		//findContours(thre_img, contours, hierarchy, CV_RETR_LIST, CHAIN_APPROX_SIMPLE);    //获取轮廓
		//draw counters
		if (isdebug)
		{
			for (int i = 0; i < contours.size(); ++i) {  //绘制所有轮廓
				cv::drawContours(show, contours, i, cv::Scalar(0, 255, 0));  //thickness为-1时为填充整个轮廓
			}
		}
		//printf("findProblScale() thrsValu = %d,find %d contours total", thrsValu,contours.size());
		//寻找外接矩形,并筛选出合适的
		//2.2.3 find the RotatedRect meet conditions. this can use multithread 
		for (int i = 0; i < contours.size(); i++)
		{
			Point2f ps[4];//外接矩形四个端点的集合
			RotatedRect box = minAreaRect(Mat(contours[i]));  //计算每个轮廓最小外接矩形

			box.points(ps);  //将最小外接矩形的四个端点复制给ps数组
			float height = box.size.height;
			float width = box.size.width;
			float rate_lw = width < height ? height / width : width / height; //calculate rate length:width
			int areasize = box.size.height*box.size.width;//面积

			if ((areasize > areasize_min)&(areasize < areasize_max)&(rate_lw_min < rate_lw&rate_lw < rate_lw_max))//condition need to meet
			{
				//check  this box whether exist in  boxSet. If exit,then skip it.
				//if (!IsExist(boxSet,box, error))
				{
					boxSet.push_back(box);
				}
				//绘制最小外接轮廓的四条边
				if (isdebug)
				{
					for (int j = 0; j < 4; j++)
					{
						line(show, Point(ps[j]), Point(ps[(j + 1) % 4]), Scalar(255, 0, 255), 2);
					}
				}
			}

		}
		if (isdebug)
		{
			namedWindow("findreslt", WINDOW_NORMAL);
			imshow("findreslt", show);
			waitKey(100);
			//waitKey(1);
		}

	}
	return 0;
}
//
vector<RotatedRect> NMS(const Mat& img, vector<RotatedRect>& boxSet, int simThres, bool isdebug)
{
	Mat NMSimg = img.clone();
	vector<RotatedRect> retboxSet;//store the result needing to return.
	
	while (!boxSet.empty())
	{
		vector<RotatedRect> simboxSet;
		RotatedRect boxtmp = boxSet[0];//move boxSet[0] to boxtmp
		boxSet.erase(boxSet.begin()); //takes 0.3ms
		simboxSet.push_back(boxtmp);
		//1.2 find similar boxs in boxSet and move them  to SimboxSet
		vector<RotatedRect>::iterator iter;
		for (iter = boxSet.begin(); iter < boxSet.end();)
		{
			if (IsRRCross(*iter, boxtmp))
			{
				RotatedRect simbox = *iter;
				boxSet.erase(iter);
				simboxSet.emplace_back(simbox);
				/*drawRotatedRect(NMSimg, simbox,Scalar(0,255,0),2);
				imshow("NMSimg", NMSimg);
				waitKey(300);*/
			}
			else
			{
				iter++;
			}
		}
		//static int i = 0;
		//i++;
		//printf("i=%d,simboxSet=%d\r", i, simboxSet.size());
		//1.2 only remain simboxSet number > simThres
		if (simboxSet.size() >= simThres)
		{
			sort(simboxSet.begin(), simboxSet.end(), [](RotatedRect& box1, RotatedRect& box2) {
				if (box1.size.area() > box2.size.area()) return true;// from big to small
				else return false;
			});
			RotatedRect retbox = simboxSet[floor(simboxSet.size() / 3)];
			retboxSet.push_back(retbox);

		}

	}
	//check again,promisewhether if there has cross boxes.
	vector<RotatedRect>::iterator iter;
	vector<RotatedRect>::iterator iter2;
	for (iter = retboxSet.begin(); iter < retboxSet.end(); iter++)
	{
		RotatedRect boxtmp = *iter;//move boxSet[0] to boxtmp
		//find similar boxs in boxSet and move them  to SimboxSet

		for (iter2 = iter + 1; iter2 < retboxSet.end();)
		{
			if (IsRRCross(boxtmp, *iter2))
			{
				retboxSet.erase(iter2);
			}
			else
			{
				iter2++;
			}
		}
	}

	if (isdebug)
	{
		DrawRotaRects(NMSimg, retboxSet);
		namedWindow("NMSimg", WINDOW_NORMAL);
		imshow("NMSimg", NMSimg);
		waitKey(1);
	}

	return retboxSet;
}
//Pick out scales and sort them
//Notice:variable 'boxSet' will be changed after calling this function. 
vector <vector<RotatedRect>> PickSortscales(Mat img, vector<RotatedRect> &boxSet, bool isdebug, int sortMethod, int minChainNum, Rect view_Rect)
{

	vector<RotatedRect> boxSet1(boxSet);
	vector<RotatedRect> boxSettmp;
	vector <vector<RotatedRect>> boxChainsSet;//store all boxes chains
	vector<RotatedRect>::iterator iter;
	//1.1 if boxSet is empty then break.
	if (boxSet.empty())
	{
		Error("boxSet is empty!");
		return boxChainsSet;
	}
	//1.2 sort by distance between origin point and box.
	if (sortMethod == 0)
	{
		sort(boxSet1.begin(), boxSet1.end(), [&](RotatedRect & box1, RotatedRect & box2) {
			float dist1 = box1.center.y;
			float dist2 = box2.center.y;
			if (dist1 <= dist2) return true;// from  small to big 
			else return false;
		});
	}
	else if (sortMethod == 1)
	{
		sort(boxSet1.begin(), boxSet1.end(), [&](RotatedRect & box1, RotatedRect & box2) {
			float dist1 = box1.center.y;
			float dist2 = box2.center.y;
			if (dist1 >= dist2) return true;// from  big to small
			else return false;
		});
	}
	//2.1 iterate boxSet1
	for (iter = boxSet1.end() - 1; !boxSet1.empty();)
	{
		
		RotatedRect rectx = *iter;
		view_Rect.x = rectx.center.x - view_Rect.width / 2;
		view_Rect.y = rectx.center.y - view_Rect.height / 2;
		//remove  Rect in view less than 2
		if (countRectsNum_InViewRect(view_Rect, boxSet) < 2)
		{
			//printf("不符合");
			if (boxSettmp.size() > 1)
			{
				boxChainsSet.push_back(boxSettmp);
			}
			boxSettmp.clear();
			boxSet1.erase(iter);
			iter = boxSet1.end() - 1;
			continue;
		}
		//printf("符合");
		if (boxSettmp.empty())
		{
			boxSettmp.push_back(rectx);
		}
		boxSet1.erase(iter);
		//find boxes in view,and find suitable box as neighbour.
		vector<vector<RotatedRect>::iterator> iterboxInView = findRects_InViewRect(view_Rect, boxSet1);
		sort(iterboxInView.begin(), iterboxInView.end(), [&](vector<RotatedRect>::iterator& iter1, vector<RotatedRect>::iterator& iter2) {
			float dist1 = getDistance(rectx.center, (*iter1).center);
			float dist2 = getDistance(rectx.center, (*iter2).center);
			if (dist1 < dist2) return true;// from  small to big 
			else return false;
		});
		int i = 0;
		//find nearest and meet position condition box as neighbour.
		for (i = 0; i < iterboxInView.size(); i++)
		{
			if (IsRRmeetPositionCondition(rectx, (*iterboxInView[i])))
			{
				if (isdebug)
				{
					Mat imgPickSortscales = img.clone();
					DrawRotaRects(imgPickSortscales, boxSet1);
					rectangle(imgPickSortscales, view_Rect, Scalar(155, 0, 100), 5);
					/*namedWindow("imgPickSortscales", WINDOW_NORMAL);
					imshow("imgPickSortscales", imgPickSortscales);*/
					drawRotatedRect(imgPickSortscales, (*iterboxInView[i]), Scalar(0, 255, 0), 2);
					namedWindow("PSscales_img", WINDOW_NORMAL);
					imshow("PSscales_img", imgPickSortscales);
					//waitKey(1000);
					waitKey(1);
				}
				//printf("meet condition ");

				boxSettmp.push_back((*iterboxInView[i]));

				iter = iterboxInView[i];
				break;
			}
		}
		// If not founded,then abandon rectx from boxSet.
		if (i == iterboxInView.size())
		{
			if (boxSettmp.size() > 1)
			{
				boxChainsSet.push_back(boxSettmp);
			}
			boxSettmp.clear();
			//printf("not meet condition ");
			iter = boxSet1.end() - 1;
		}

	}

	Mat remainedBoxesChain = img.clone();
	for (auto it = boxChainsSet.begin(); it < boxChainsSet.end();)
	{
		//abandon size less than minChainNum
		if ((*it).size() < minChainNum)
		{
			boxChainsSet.erase(it); continue;
		}
		// whether show remained scales chain or not
		if (isdebug)
		{
			DrawRotaRects(remainedBoxesChain, (*it));
			namedWindow("remainedBoxesChain", WINDOW_NORMAL);
			imshow("remainedBoxesChain", remainedBoxesChain);
			//waitKey(1000);
			waitKey(1);
		}
		it++;
		
	}
	return boxChainsSet;
}

vector<RotatedRect> rmSimScale(Mat img, vector<RotatedRect> &boxSet)
{
	Mat drboard(img.size(), CV_8UC1, Scalar(0));
	Mat gray;
	vector<RotatedRect>::iterator iter;
	for (iter = boxSet.begin(); iter < boxSet.end(); iter++)
	{
		drawRotatedRect(drboard, *iter, Scalar(255), 2, 8, 0, 1);
	}
	//寻找轮廓
	vector<vector<Point>> contours2;    //储存轮廓
	vector<Vec4i> hierarchy2;
	findContours(drboard, contours2, hierarchy2, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);    //获取轮廓
	for (int i = 0; i < contours2.size(); ++i) {  //绘制所有轮廓
		cv::drawContours(drboard, contours2, i, cv::Scalar(100), 4);  //thickness为-1时为填充整个轮廓
	}
	vector<RotatedRect> retbox(contours2.size()); //定义最小外接矩形集合

	vector<vector<Point> > squares2;
	for (int i = 0; i < contours2.size(); i++)
	{


		retbox[i] = minAreaRect(Mat(contours2[i]));  //计算每个轮廓最小外接矩形
	}
	imshow("drboard", drboard);
	waitKey(1);
	return retbox;

}