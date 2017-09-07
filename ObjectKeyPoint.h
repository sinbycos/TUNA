
#pragma once

#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include "LBSP.h"

using namespace std;
using namespace cv;


struct objectKeys
{
	cv::KeyPoint key;
	cv::Point2f dis_Cen, predC;
	vector<cv::Point2f> predCenters;
	float distance;
	float fPairdistance;
	float ori;
	float weight1;
	float weight2;
	float proxFactor;
	float jitterFactor;
	float diffT;
	float diffT1;
	float fXDis;
	float fYDis;
	cv::Mat descriptor;
	int index;
	int indi;
	int nTimes;
	float fLBSPgradient;
	float fGradChange;
	int setAM;
	float distFromOri;
	Point2f spatialVec;
	int Oct;
	
};


class ObjectKeyPoint : public cv::KeyPoint{


public:

//! full constructor
	ObjectKeyPoint();

//! default destructor
	virtual ~ObjectKeyPoint();




//! Descriptor
	cv::Mat computeDes(const cv::Mat& oInitImg, std::vector<objectKeys>& m_voFilteredKeyPoints); 


//! Keypoints
	std::vector<objectKeys> m_voKeyPoints;


//! Filtered keypoints	
	std::vector<objectKeys> m_voFilteredKeyPoints, m_voBGKeyPoints, m_voFilteredKeyPoints1, m_voFilteredKeyPointsTrackROI, matchedKeysInTrackROI;


//! KeyPoint Descriptors
	cv::Mat m_voDescriptors;



	cv::Rect m_voROIPos;

	unsigned int weight;
	unsigned int vote;


std::vector<unsigned int> m_voIndicator;


void filteredKeyPoints(std::vector<objectKeys>& m_voKeyPoints, std::vector<objectKeys>& m_voFilteredKeyPoints, cv::Rect ROI, int contextFlag);


void filterBGKeyPoints(std::vector<objectKeys>& m_voKeyPoints, std::vector<objectKeys>& m_voBGKeyPoints, cv::Rect ROI, float fWeight);

std::vector<DMatch> sortMatches(vector<DMatch>& matches);

cv::Point2f getPosMatchKeys(vector<DMatch>& finalMatches2, std::vector<KeyPoint>& keyPoints1);

Point getCenter(vector<objectKeys> keyPoints);

double getDis(vector<objectKeys> keyPoints);

void encodeStructure(std::vector<objectKeys>& m_voFilteredKeyPoints, cv::Point2f center, cv::Mat image2, float& fUpdate);

void voting(std::vector<objectKeys>& m_voFilteredKeyPoints, std::vector<objectKeys>& keyPointsROI, std::vector<objectKeys>& m_voKeyPoints, std::vector<Point2f>& accum, cv::Mat image2, cv::Rect ROI, cv::Point2f& previousCenter, cv::Point2f& predCenter, int& frameNum, float& fExpo,
	float& distBetCenters, Mat oColor1, Mat oColor2, double& fCompareHS);

vector<double> getDiff(Point cen, double rad, cv::Point pt1, cv::Point pt2);

void newPoints(vector<double> diff, Point cen, double rad,cv::Point pt1, cv::Point pt2, cv::Rect ROI);

bool featureMatchedOnlyOnce( vector< DMatch > matches, int query_index, int train_index );

cv::Point2f ObjectKeyPoint::rotate(cv::Point2f p, float rad);

vector<vector<DMatch>> searchRegion(cv::Mat descriptors1, cv::Mat descriptors2, vector<KeyPoint> keyPoints1, vector<vector<DMatch>> matchesR, cv::Mat image, cv::BFMatcher bf);

void keysFromCenter(vector<KeyPoint> keyPointsImg, Point2f& center, Point2f& pt1, Point2f& pt2, cv::Rect& ROI);

static bool sortDisCompare(const pair<KeyPoint,Point2f>& A, const pair<KeyPoint,Point2f>& B ); // STATIC SO THAT THE ELEMENTS DO NOT GET MODIFIED IN A CLASS, ELSE A POINTER HAVE TO BE PROVIDED


void getMinMaxWeightKeyPoint(vector<objectKeys>& keyPoint1, vector<objectKeys>& keyPoint2, std::vector<int>& voIndexes);




void predictCenter(std::vector< std::vector<float> > accum, float tRadius, std::vector<objectKeys>& m_voFilteredKeyPoints);

void update(std::vector<objectKeys>& m_voFilteredKeyPoints, std::vector<objectKeys>& m_voKeyPoints, cv::Point2f& predCenter, float& a, int& frame, float& fUpdate, float& fRateRemove, float& fCov, float& fSigma);

void removeKeys(std::vector<objectKeys>& m_voFilteredKeyPoints, float& tWeight, cv::Point2f& predCenter);

void addKeys(std::vector<objectKeys>& m_voFilteredKeyPointsTrackROI, std::vector<objectKeys>& m_voFilteredKeyPoints, std::vector<objectKeys>& keyPointsROI, cv::Rect ROI, cv::Point2f& predCenter, float& fUpdate, float& fnewWeight);
 
void filterMatches(vector<vector<DMatch>>& foundMatches);





void weightedKeysinTrackROI(std::vector<objectKeys>& oKeyPoints, std::vector<objectKeys>& oKeyPoints1, cv::Rect ROI, float& fWeight);
void nonModelKeysinTrackROI(std::vector<objectKeys>& oKeyPoints, std::vector<objectKeys>& oKeyPoints1, cv::Rect ROI);
void modelKeysinTrackROI(std::vector<objectKeys>& oKeyPoints, std::vector<objectKeys>& oKeyPoints1, cv::Rect ROI);

void modelMatchedKeysinTrackROI(std::vector<objectKeys>& oKeyPoints, std::vector<objectKeys>& oKeyPoints1, cv::Rect ROI);

cv::Rect readGT(string& gtFile);

cv::Rect readGTCenter(string& gtFile);

void splitFilename (const string& str);

void matchedAMKeyPoints(cv::Rect trackROI, cv::Point2f predCenter, std::vector<objectKeys> m_voKeyPoints );

void setAMzero(std::vector<objectKeys>& o_vKP);
void calLBSPHist(vector<ushort>& gradientValue, cv::Mat& oDescriptor, cv::Mat& oLBSPHist, int& nKeys);
void calWeightedColorHist(cv::Mat& oImage, cv::Mat& oTest, Point2f& pt1, Point2f& pt2, cv::Mat& oHist, cv::Rect& oROI);
void calWeightedColor(cv::Mat& oImage, cv::Mat& oHist, cv::Rect& oROI);


void setLBSPROIModel(cv::Mat& oImage, Point2f& pt1, Point2f& pt2, cv::Rect& oLBSPROI);
void setLBSPROINonModel(cv::Mat& oImage, Point2f& pt1, Point2f& pt2, cv::Rect& oLBSPROI);
void ROIAdjust(const cv::Mat& oSource, Point2f& pt1, Point2f& pt2);
void ROIBoxAdjust(const cv::Mat& oSource, cv::Rect& oBox);

void updateTemplateModel(cv::Mat& oModel1, cv::Mat& oImage1, cv::Mat& oModel2, cv::Mat& oImage2, cv::Mat& oHSModel, cv::Mat& oHSNonModel);

void computePairDistance(std::vector<objectKeys>& keyPoint1, std::vector<objectKeys>& keyPoint2, std::vector<int>& voIndexes, std::vector<Point2f>& voDist1, std::vector<Point2f>& voDist2);

void detectScaleChange(std::vector<Point2f>& voDist1, std::vector<Point2f>& voDist2, float& scaleChange, float& scaleChange2);



inline static void matchLBSPDes(cv::Mat & oDesc1, cv::Mat& oDesc2, vector<int>& vOutArray)
{

	for (auto i = 0; i < oDesc1.rows; i++)
	{
		int nRes = hdist(oDesc1.at<ushort>(i, 0) , oDesc2.at<ushort>(i, 0));
		vOutArray.push_back(nRes);
	}

}

};
