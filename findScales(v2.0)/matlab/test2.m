%mex -v COMPFLAGS='$COMPFLAGS -std=c++17' findProblScale.cpp LineBasic.cpp '-ID:\software\opencv-4.6\opencv\build\include' -LD:\software\opencv-4.6\opencv\build\x64\vc15\lib '-lopencv_world460'


clear all;close all;
img = imread('1.bmp');
img_out = findProblScale(img);

figure;imshow(uint8(img_out));