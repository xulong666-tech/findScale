#include "LineBasic.h"
#include <iostream>
#include "mex.h"
#include <cassert>
using namespace cv;

// 灰度图和彩色图结合接口函数
//nlhs 输出参数个数，plhs 输出参数指针，nrhs，输入参数个数，prhs输入参数指针（输出和输入参数的操作都通过指针的方式进行）。
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
// 	float sigma_s = mxGetScalar(prhs[1]);
// 	float sigma_r = mxGetScalar(prhs[2]);
// 	int iteration = mxGetScalar(prhs[3]);
// 
	uchar* inData;
	uchar* InCurRow;

	inData = (uchar*)mxGetPr(prhs[0]);////获得指向输入矩阵的指针
    int rows = mxGetM(prhs[0]); //获得输入矩阵的行数
    int cols = mxGetN(prhs[0]); //获得输入矩阵的列数
    int channel = mxGetNumberOfDimensions(prhs[0]);
    cols = cols/channel;
    mexPrintf("rows is %i\n", rows);
    Mat image(rows,cols,CV_8UC3);//CV_8UC3
    for (int i = 0; i < rows; i++)
    {
        InCurRow = (uchar*)image.ptr<uchar>(i);//获取第i行的指针
        for (int j = 0; j < cols; j++)  
        {
            for (int k = 0; k < channel; k++)
            {
                   image.at<Vec3b>(i, j)[2 - k] = *(inData + i + j * rows + k * rows * cols);
                   InCurRow[j * channel + (2 - k)] = *(inData + i + j * rows + k * rows * cols);
                  *(InCurRow + j * channel + (2 - k)) = *(inData + i + j * rows + k * rows * cols);
            }
        }
    }
    
    std::vector<RotatedRect> probboxs;
    char *condition[4] = { (char*)"30",(char*)"1700",(char*)"1.5",(char*)"8" };
    findProblScale(image, probboxs, 20, 200, 4, condition, 1);
    
	Mat out(rows,cols,CV_8UC3);//CV_8UC3
	
    out = image;
	//cvtColor(image, out, CV_RGB2HSV);//转换为灰度图
   /* plhs[0] = mxCreateDoubleMatrix(rows, cols, mxCOMPLEX);
    double *outData_R = mxGetPr(plhs[0]); 
    plhs[1] = mxCreateDoubleMatrix(rows, cols, mxREAL);  
    double *outData_G = mxGetPr(plhs[1]); 
    plhs[2] = mxCreateDoubleMatrix(rows, cols, mxREAL);  
    double *outData_B = mxGetPr(plhs[2]); 

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			*(outData_R + i + j * rows) = (double)out.at<Vec3b>(i, j)[2];
			*(outData_G + i + j * rows) = (double)out.at<Vec3b>(i, j)[1];
			*(outData_B + i + j * rows) = (double)out.at<Vec3b>(i, j)[0];
		}
	} */
	const int DIMEN = 3;
	int height = rows;
	int width = cols;
	int channels = 3;

	mwSize dims[DIMEN] = { height, width, channels };

	plhs[0] = mxCreateNumericArray(DIMEN, dims, mxUINT8_CLASS, mxREAL);
	inData = (uchar*)mxGetPr(plhs[0]);////获得指向输入矩阵的指针

	for (int i = 0; i < rows; i++)
	{
		InCurRow = (uchar*)out.ptr<uchar>(i);//获取第i行的指针
		for (int j = 0; j < cols; j++)
		{
			for (int k = 0; k < channel; k++)
			{
				*(inData + i + j * rows + k * rows * cols) = out.at<Vec3b>(i, j)[2 - k];
				*(inData + i + j * rows + k * rows * cols)= InCurRow[j * channel + (2 - k)];
				*(inData + i + j * rows + k * rows * cols) = *(InCurRow + j * channel + (2 - k));
			}
		}
	}
    //printf("xulong666");
    printf("nlhs=%d\n",nlhs);
}