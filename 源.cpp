#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <videoio_c.h>
#include<string>
#include<io.h>
#include<list>
#include<array>

using namespace std;
using namespace cv;
#define NUM_FRAME 300
#define SIZE 7

char path[100];//输入文件路径

struct shot
{
	list<array<float, 22> >content;
	list<int> id;
	array<float, 22> center;
};


float similarity(array<float, 22>  x1, array<float, 22>  x2)
{
	float s1 = 0, s2 = 0, s3 = 0;
	float alpha1 = 0.5, alpha2 = 0.3, alpha3 = 0.2;
	for (int i = 0; i < 12; i++) { //计算三个颜色直方图的相似性：累加两张图像直方图相同索引处对应的最小值
		s1 += min(x1[i], x2[i]);
	}
	for (int i = 12; i < 17; i++) {
		s2 += min(x1[i], x2[i]);
	}
	for (int i = 17; i < 22; i++) {
		s3 += min(x1[i], x2[i]);
	}
	return s1 * alpha1 + s2 * alpha2 + s3 * alpha3;
}

int findMaxEntropyId(list<array<float, 22> >x, list<int> y)
{
	float s1, s2, s3, max;

	list<array<float, 22> >::iterator it;
	list<int>::iterator i = y.begin();
	int id = 0;

	for (it = x.begin(); it != x.end(); it++, i++) {
		s1 = 0.0f, s2 = 0.0f, s3 = 0.0f, max = 0.0f;
		for (int j = 0; j < 12; j++) {
			if ((*it)[j] != 0)s1 += -(*it)[j] * log((*it)[j]) / log(2);
		}
		for (int j = 12; j < 17; j++) {
			if ((*it)[j] != 0)s2 += -(*it)[j] * log((*it)[j]) / log(2);
		}
		for (int j = 17; j < 22; j++) {
			if ((*it)[j] != 0)s3 += -(*it)[j] * log((*it)[j]) / log(2);
		}
		float s = 0.5f * s1 + 0.3f * s2 + 0.2f * s3;
		//printf("s = %f\n", s);
		if (s > max) {
			max = s;
			id = *i;
		}
	}
	return id;
}

const array<float, 22> operator +(const array<float, 22>& x, const array<float, 22>& y)
{
	array<float, 22>ans;
	for (int i = 0; i < 22; i++) {
		ans[i] = x[i] + y[i];
	}
	return ans;
}

const array<float, 22> operator /(const array<float, 22>& x, int s)
{
	array<float, 22>ans;
	for (int i = 0; i < 22; i++) {
		ans[i] = x[i] / s;
	}
	return ans;
}

void combine(vector<shot>& Shot, int i, int j)
{
	list<array<float, 22> >::iterator it;
	list<int>::iterator k = Shot[j].id.begin();
	vector<shot>::iterator v = Shot.begin() + j;
	for (it = Shot[j].content.begin(); it != Shot[j].content.end(); it++, k++) { //内容合并
		Shot[i].content.push_back(*it);
		Shot[i].center = *it + Shot[i].center; //加在一起，非首尾相连
		Shot[i].id.push_back(*k);
	}
	Shot.erase(v); //删除后一个
}

array<float, 22> sum(list<array<float, 22> >& arr) //返回为content中的所有array合为的一个
{
	array<float, 22> ans = { 0 };
	list<array<float, 22> >::iterator it;
	for (it = arr.begin(); it != arr.end(); it++) {
		for (int i = 0; i < 22; i++) {
			ans[i] += (*it)[i];
		}
	}
	return ans;
}

void mergeArray(int a[], int first, int mid, int last, int temp[]) {
	int i = first;
	int j = mid + 1;
	int m = mid;
	int n = last;
	int k = 0;

	while (i <= m && j <= n) {
		if (a[i] <= a[j]) temp[k++] = a[i++];
		else temp[k++] = a[j++];
	}
	while (i <= m) temp[k++] = a[i++];
	while (j <= n) temp[k++] = a[j++];

	for (i = 0; i < k; i++)
		a[first + i] = temp[i];
}

void mergeSort(int a[], int first, int last, int temp[]) {
	if (first < last) {
		int mid = (first + last) / 2;
		mergeSort(a, first, mid, temp);
		mergeSort(a, mid + 1, last, temp);
		mergeArray(a, first, mid, last, temp);
	}
}

//将图片序列转换为视频
void handleVideo()
{
	int i = 0;
	IplImage* img = 0;//读入图像
	IplImage* outimg = 0;//修改图像尺寸
	char image_name[100];//图像名字
	char videoname[100];
	strcpy_s(videoname, "C:\\Users\\lenovo\\Desktop\\22.avi");

	//从文件读入视频
	CvCapture* capture = cvCreateFileCapture(videoname);
	//读取和显示
	IplImage* frameimg;//从视频中提取的帧图像
	int fps = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FPS);//视频的fps
	int frameH = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);//视频的高度
	int frameW = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);//视频的宽度
	printf("\tvideo height : %d\n\tvideo width : %d\n\tfps : %d\n", frameH, frameW, fps);

	list<array<float, 22> >colorbar;

	//创建窗口  
	cvNamedWindow("mainWin", CV_WINDOW_AUTOSIZE);
	//读入图片，并制作幻灯片
	while (1)
	{
		frameimg = cvQueryFrame(capture); //获取一帧图片
		if (!frameimg)break;//读到尽头，退出
		cvShowImage("mainWin", frameimg); //显示
        char c = cvWaitKey(3); //延迟
        if (c == 'c') break;
		cvCvtColor(frameimg, frameimg, CV_BGR2HSV); //将一帧图像转为HSV空间
		array<float, 22> color = { 0 }; //当前帧的hsv
		uchar* data = (uchar*)frameimg->imageData; //imageData就是一个指针，指向某张图片像素值数据的首地址
		int step = frameimg->widthStep / sizeof(uchar); //widthStep表示存储一行像素需要的字节数，为4的倍数
		int channels = frameimg->nChannels;
		uchar* h = new uchar[frameimg->height * frameimg->width]; //数组，大小为整张图的像素点数量
		uchar* s = new uchar[frameimg->height * frameimg->width];
		uchar* v = new uchar[frameimg->height * frameimg->width];
		for (int i = 0; i < frameimg->height; i++) {
			for (int j = 0; j < frameimg->width; j++) {
				h[i * frameimg->height + j] = data[i * step + j * channels + 0] / 15;
				if (h[i * frameimg->height + j] > 11)h[i * frameimg->height + j] = 11;
				s[i * frameimg->height + j] = data[i * step + j * channels + 1] / 51;
				if (s[i * frameimg->height + j] > 4)s[i * frameimg->height + j] = 4;
				v[i * frameimg->height + j] = data[i * step + j * channels + 2] / 51;
				if (v[i * frameimg->height + j] > 4)v[i * frameimg->height + j] = 4;

				color[h[i * frameimg->height + j]]++; //0-11
				color[12 + s[i * frameimg->height + j]]++; //12-17
				color[17 + v[i * frameimg->height + j]]++; //17-22
			}
		}
		for (int i = 0; i < 22; i++) {
			color[i] /= frameimg->height * frameimg->width;
			//printf("%lf\n", color[i]);
		}
		colorbar.push_back(color); //得到每一帧的color
	}

	printf("\t共%d帧\n", colorbar.size());

	float threshold = 0.94f;
	list<array<float, 22> >::iterator it = colorbar.begin();
	it++; //从第二帧开始
	vector<shot>Shot; //所有聚类

	//放入第一帧
	shot first;
	first.content.push_back(*colorbar.begin());
	first.center = *colorbar.begin();
	first.id.push_back(0);
	Shot.push_back(first); //第一帧默认为一个聚类

	printf("1--%d--%d\n", Shot[0].id.front(), Shot[0].id.back());

	int count = 0; //帧序号
	int num = 1; //聚类数目
	int index = 0;
	float max = 0;
	for (; it != colorbar.end(); it++) { //所有帧图像的hsv遍历
		max = 0;
		index = 0;
		//计算相似度最大的
		for (int i = 0; i < num; i++) {
			float ratio = similarity(*it, Shot[i].center); //第二帧（*it）与第一帧比较相似度（通过聚类中心）
			if (ratio > max) {
				max = ratio; //得到最大相似度
				index = i; //记录当前聚类
			}
		}
		//如果最大的小于某个阈值，则新建一个聚类
		if (max < threshold) {
			num++; //聚类数目增加
			shot newshot;
			newshot.center = *it; //得到这一帧的color[]
			newshot.content.push_back(*it);
			newshot.id.push_back(count);
			Shot.push_back(newshot); //加入新聚类
		}
		else {
			Shot[index].center = (*it + sum(Shot[index].content)) / (Shot[index].content.size() + 1); //重新计算聚类中心
			Shot[index].content.push_back(*it); //加入当前聚类
			Shot[index].id.push_back(count);//记录为第几帧
			if (index == 0) printf("**%d\n", Shot[index].id.back());
		}
		count++;
	}

	printf("2--%d--%d\n", Shot[1].id.front(), Shot[1].id.back());

	for (int i = 0; i < Shot.size(); i++) { //合并小聚类
		if (Shot[i].content.size() < 10 && i > 0) {
			combine(Shot, i - 1, i);
			i--;
		}
	}
	float maxE = 0.0f;
	int indexE = 0;
	int final[999];
	for (int i = 0; i < Shot.size(); i++) {
		int id = findMaxEntropyId(Shot[i].content, Shot[i].id);
		//printf("%d\n", id);
		final[i] = id;
	}
	printf("%d", Shot.size());
	cvReleaseCapture(&capture);
	

	CvCapture* capture2 = cvCreateFileCapture(videoname);
	IplImage* frameimg2;
	int number = 0;
	int f = 0;
	sort(final, final + Shot.size());
	for (int i = 0; i < Shot.size(); i++)
		printf("\n%d***%d", i, final[i]);
	while (1) {
		frameimg2 = cvQueryFrame(capture2); //获取一帧图片
		if (!frameimg2)break;
		if (final[f] == number) {
			char str[200];
			snprintf(str, 100, "%s%d%s", "C:\\Users\\lenovo\\Desktop\\Project1\\KeyFrame2\\第", number, "帧.jpg");
			cvSaveImage(str, frameimg2); //保存
			if(f < Shot.size())
				f++;
			else break;
		}
		number++;
	}
	cvReleaseCapture(&capture2);
	cvDestroyWindow("mainWin");
}


int main(int argc, char* argv[])
{
	handleVideo();
	waitKey();
	system("pause");
	return 0;
}


//#include "opencv2/core.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/imgproc.hpp"
//#include "opencv2/videoio.hpp"
//
//using namespace std;
//using namespace cv;
//
//int main()
//{
//    char videoname[100];
//	strcpy_s(videoname, "C:\\Users\\lenovo\\Desktop\\vtest_Xvid.avi");
//    cvNamedWindow("WIND0W", CV_WINDOW_AUTOSIZE);
//    CvCapture* capture = cvCreateFileCapture(videoname);
//    IplImage* frame;
//
//    while (1)
//    {
//        frame = cvQueryFrame(capture);
//        if (!frame) break;
//        cvShowImage("WIND0", frame);
//        char c = cvWaitKey(33);
//        if (c == 'c') break;
//    }
//
//    cvReleaseCapture(&capture);
//    cvDestroyWindow("WIND0");
//
//
//    return 0;
//}