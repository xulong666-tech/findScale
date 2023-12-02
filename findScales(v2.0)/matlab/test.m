% opencvIncludePath = 'D:\software\opencv-4.6\opencv\build\include';
% opencvlib = 'D:\software\opencv-4.6\opencv\build\x64\vc15\lib\opencv_world460.lib';
% ipath = ['-I' opencvIncludePath]
%mex('-v','-R2017b',ipath,'findProblScale.cpp',opencvlib);
mex -v COMPFLAGS='$COMPFLAGS -std=c++17' findProblScale.cpp  '-ID:\software\opencv-4.6\opencv\build\include' -LD:\software\opencv-4.6\opencv\build\x64\vc15\lib '-lopencv_world460'


