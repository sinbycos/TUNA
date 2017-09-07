#include "boost/lambda/lambda.hpp"
#include "boost/filesystem.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/algorithm/string/find_iterator.hpp"
#include "boost/regex.hpp"
#include <iostream>
#include <iterator>
#include <algorithm>
#include <vector>
#include <numeric>
#include <cmath>
#include<math.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"


#include <stdio.h>
#include <fstream>
#include <string>


#define DRAW 1
#define LB 0 
#define CENTER 0
#define SAVE 1
#define THRESH 10



#include "ObjectKeyPoint.h"

#include "LBSP.h"
#include "DistanceUtils.h"



using namespace std;
using namespace cv;
using namespace boost;
using namespace boost::filesystem;

ObjectKeyPoint oKAlgModel, oKAlgNonModel;


const char* keys = {
	"{c  |context			|0			| use context or not}"
	"{t  |radius			|250		| radiusMatch }"
	"{a  |learningRate		|0.1		| learningRate }"
	"{z  |lRateRemove		|0.1		| RateRemove }"
	"{b  |sigma				|200.0		| fExpo }"
	"{u  |update			|0.05		| update }"
	"{s  |seqName			|		    | sequence Name }"
	"{p  |seqPath			|           | Image directory  directory file path}"
	"{r  |path				|			| output files    }"
	"{f  |start frame		|1			| start frame }"
	"{e  |end frame			|			| end frame }"
	"{n  |nz				|  nz		| nz }"
	"{ext|					|           |extension input images}"
	"{x  |top left			|           | BB x}"
	"{y  |top left          |           | BB y}"
	"{w  |width		        |           | BB width}"
	"{h  |height            |           | BB height}"
};

int main(int argc, char** argv)

{
	ofstream file1;

	int nTotalMatches = 0;
	
	CommandLineParser parser(argc, argv, keys);


	
	int nContext = parser.get<int>("c");
	int nRadius = parser.get<int>("t");
	float fLearningRate = parser.get<float>("a");
	float fRateRemove = parser.get<float>("z");
	float fExpo = parser.get<float>("b");
	float fUpdate = parser.get<float>("u");
	const string sSeqName = parser.get<String>("s");
	const string sSeqPath = parser.get<String>("p");


	const string sResultDir = parser.get<String>("r");
	const int nStartFrame = parser.get<int>("f");
	const  int nEndFrame = parser.get<int>("e");
	int nNz = parser.get<int>("n");
	const string sSeqExt = parser.get<String>("ext");
	int nX = parser.get<int>("x");
	int nY = parser.get<int>("y");
	int nW = parser.get<int>("w");
	int nH = parser.get<int>("h");



	int iMaxWeightedKeyIndexSec= 0;
	int iMaxWeightedKeyIndexFirst = 0;
	int iMinWeightedKeyIndexSec = 0;

	int iMinWeightedKeyIndexFirst = 0;



	int iSecIndexMin = 0;
	int iThirdIndexMin = 0;

	vector<string> vidDir;
	vidDir.push_back(sSeqName);
	string vidFile;
	string gtFile;

	regex pattern("(.*\\.jpg)");
	//boost::regex pattern("(.*\\.png)");
	//boost::regex pattern("(.*\\.bmp)");
	vector<std::string> accumulator;

	vector<std::string>::iterator iter1;

	


	Mat oImage1, oImage2, oDescriptor1, oDescriptor2, oDescriptorLBSPModel, oDescriptorLBSPNonModel, oColorModelHist, oColorNonModelHist,oMatches,oMatches1, oMatches2;
	Mat oLBSPModelHist, oLBSPNonModelHist;
	Mat oModelR, oModelG, oModelB, oNonModelR, oNonModelG, oNonModelB, oHueColor, oLBSPColor, oLBSPColor1, oLBSPColor2, oHueColor2;
	vector<objectKeys> voKeyPoints1, voKeyPoints2, voKeyPointsROI, voModelKeys, voNonModelKeys;
	vector<vector<DMatch>> matchesR, matchesR1;
	
	Mat oHSModel, oHSNonModel, oHSModelT, oHSNonModelT, oColor1, oColor2;
	const int nSizes[3] = { 16, 16,16 };

	oColor1 = Mat::zeros(3, nSizes, CV_32FC1);
	oColor2 = Mat::zeros(3, nSizes, CV_32FC1);
	vector<Mat> voBFTraining;
	vector<KeyPoint> voDrawKeysModel, voDrawKeysNonModel;
	
	double fCompareHistLBSP, fCompareHS, fLBSPThresh, fHSThresh ;
	
	Rect ROI, BackROI, oLBSPROIModel, oLBSPROINonModel;
	Point2f pt1, pt2, BPt1, BPt2;
	
	Point2f previousCenter, predCenter, center, BGcenter;

	int nFrameCounter = 0;
	float fThreshWeight = 0.1, fBGweight = 30;
	float fdisBCenter = 0.0;
	float fnewWeight = 0.15;

	cout << "CONTEXT \t " << nContext << endl;
	cout << "BR RADIUS \t" << nRadius << endl;
	cout << "LR \t" << fLearningRate << endl;
	cout << " Gauss Denum Voting \t" << fExpo << endl;
	cout << "Const for weight \t" << fUpdate << endl;
	cout << "THREh weight \t" << fThreshWeight << endl;
	cout << " new weight added \t " << fnewWeight << "\n"<< endl;


	float fScaleChange = 0.0;
	float fFinalScaleChange = 1.0;
	float fMeanScaleChange = 0.0;
	float fCov=1.0, fSigma = 1.0;

	Point2f pairedDist1, pairedDistNew1, pairedDist2, pairedDistNew2, pairedDist3, pairedDistNew3;

	std::vector<int> voIndexes; std::vector<Point2f> voDistance1, voDistance2; std::vector<float> voAccScales;

	


	VideoWriter writer;

	Ptr<xfeatures2d::SIFT> pSift = xfeatures2d::SiftFeatureDetector::create(0, 3, 0.04, 10, 1.6); // features detected upto 3 octave layers means 3 scales

	
	
	
	BFMatcher bf(NORM_L2, true);
	
	Mat oGrey, oGrey2;


	for (vector<string>::const_iterator i = vidDir.begin(); i != vidDir.end(); ++i)
	{
		


		Mat oFramePutText1, oFramePutText2, oFramePutText3, oImageCopy1, oImageCopy2;
		int nFrame;
		const string sImageDir = sSeqPath;



		

		if (nContext == 1)
		{

		file1.open(sResultDir + "\\" + *i + "_TUNA_CONTEXT.txt ", ios::out);

		}


		if (nContext == 0)

		{

		file1.open(sResultDir + "\\" + *i + "_TUNA_NO_CONTEXT.txt ", ios::out);

		}

		path target_path (sImageDir);


		for (directory_iterator iter(target_path),end; iter!= end; ++iter)
		{



		string sImgNum = iter->path().leaf().string();

		string imgFile = sSeqPath  + sImgNum;


		if (regex_match(sImgNum,pattern))
		{


		accumulator.push_back(imgFile);


		}


		}

		
#if SAVE == 2
		file1 << "data" << "[";

#endif	

	for (iter1 = (accumulator.begin() + (nStartFrame - 1)); iter1 != (accumulator.begin() + (nEndFrame)); ++iter1) {

		

		
			if (iter1 == (accumulator.begin() + (nStartFrame - 1))){
			
		
				nFrame = nStartFrame;

				oImage1 = imread(*iter1);

				
				oFramePutText1 = oImage1.clone();
				oFramePutText3 = oImage1.clone();
				oImageCopy1 = oImage1.clone();
				oLBSPColor = oImage1.clone();

				std::vector<KeyPoint> voDetectedKeys;
				pSift->detect(oImage1, voDetectedKeys);

				char frameString[10];
				char sym[2] = "#";
				itoa(nFrame, frameString, 10);
				strcat(frameString, sym);
				cv::putText(oFramePutText1, frameString, cv::Point(10, 20), FONT_HERSHEY_COMPLEX_SMALL, 1, CV_RGB(255, 0, 0));
				voKeyPoints1.resize(voDetectedKeys.size());

				for (auto i = 0; i < voDetectedKeys.size(); ++i){

					voKeyPoints1[i].key = voDetectedKeys[i];

					voKeyPoints1[i].descriptor = Mat(1, 128, CV_32FC1);
					voKeyPoints1[i].proxFactor = -1;
					voKeyPoints1[i].index = -1;
					voKeyPoints1[i].indi = 1;

				}

				voDetectedKeys.clear();

				unsigned int i = 0;


#if CENTER == 1

				//ROI = oKAlgModel.readGTCenter(gtFile);


				center.x = ROI.x;
				center.y = ROI.y;
				pt1 = Point2f(center.x - ROI.width / 2, center.y - ROI.height / 2);


#endif

#if CENTER == 0			

				//ROI = oKAlgModel.readGTCenter(gtFile);
				ROI.x = nX;
				ROI.y = nY;
				ROI.width = nW;
				ROI.height = nH;
				
				center.x = ROI.x + ROI.width / 2;
				center.y = ROI.y + ROI.height / 2;
				pt1.x = ROI.x;
				pt1.y = ROI.y;

#endif





				if (nContext == 1)
				{

					pt2 = Point2f((center.x + ROI.width / 2), ((1.2*center.y) + (ROI.height / 2)));
				}

				if (nContext == 0)
				{
					pt2 = Point2f((center.x + ROI.width / 2), ((center.y) + (ROI.height / 2)));
				}



				


#if DRAW == 11


				rectangle(oFramePutText1, pt1, pt2, cv::Scalar(0, 255, 0), 3, 8, 0);

				namedWindow("ROI", WINDOW_AUTOSIZE);
				//moveWindow("ROI", 60, 200);
				imshow("ROI", oFramePutText1);


				cvWaitKey(10);

#endif


#if DRAW == 11


				rectangle(oFramePutText3, BPt1, BPt2, cv::Scalar(0, 255, 0), 3, 8, 0);

				namedWindow("BROI", WINDOW_AUTOSIZE);
				//moveWindow("ROI", 60, 200);
				imshow("BROI", oFramePutText3);


				cvWaitKey(10);

#endif



# if SAVE == 3
				namedWindow("TEST", WINDOW_AUTOSIZE);
				//imwrite(".\\fdOut\\context.jpg", test1);
				imshow("TEST", test1);

				cvWaitKey(10);
#endif


				



# if SAVE == 1
				if (file1.is_open())

				{


					file1 << floor(pt1.x) << "," << floor(pt1.y) << "," << (ROI.width) << "," << (ROI.height) << endl;
				}




				if (!file1.is_open()){

					if (nContext == 1)
					{

						file1.open(sResultDir + "\\" + sSeqName + "_TUNA_CONTEXT.txt ", ios::out);

					}


					if (nContext == 0)

					{

						file1.open(sResultDir + "\\" + sSeqName + "_TUNA_NO_CONTEXT.txt ", ios::out);

					}



				}


#endif					





# if SAVE == 3
				if (file1.isOpened())
				{

					file1 << "{" << "frame " << frame;
					file1 << "X " << pt1.x;
					file1 << "Y " << pt1.y;
					file1 << "W " << ROI.width;
					file1 << "H " << ROI.height << "}";
				}

#endif
# if SAVE == 2
				if (file1.isOpened())
				{

					file1 << "{" << "frame " << frame;
					file1 << "CX " << center.x;
					file1 << "CY " << center.y;
					file1 << "ROIW " << ROI.width;
					file1 << "ROIH " << ROI.height << "}";
				}

#endif			



				oKAlgModel.m_voKeyPoints = voKeyPoints1;




				oKAlgModel.filteredKeyPoints(oKAlgModel.m_voKeyPoints, oKAlgModel.m_voFilteredKeyPoints, ROI, nContext);



				oKAlgModel.encodeStructure(oKAlgModel.m_voFilteredKeyPoints, center, oImageCopy1, fUpdate); //HOW MUCH KPS HAVE MOVED FROM CENTER AND THEIR DIRECTION TOWARDS THE CENTER


				

				



		
				

				oDescriptor1 = oKAlgModel.computeDes(oImage1, oKAlgModel.m_voFilteredKeyPoints);

				

				for (auto i = 0; i < oDescriptor1.rows; ++i){
					{

						oKAlgModel.m_voFilteredKeyPoints[i].descriptor = oDescriptor1.row(i).clone();

					}
				}


		
				oKAlgModel.setLBSPROIModel(oLBSPColor, pt1, pt2, oLBSPROIModel );



				cv::cvtColor(oLBSPColor, oGrey, CV_RGB2GRAY);

				
				ushort uLBSP;  vector<ushort> voLBSP;
				
				for (auto j = 2; j < (oGrey.rows - 2); j++)
				{
					for (auto i = 2; i < (oGrey.cols - 2); i++)

					{
						
						LBSP::computeGrayscaleDescriptor(oGrey, oGrey.at<uchar>(j, i), i, j, THRESH, uLBSP);
						
						voLBSP.push_back(uLBSP);

					}


				}




				oDescriptorLBSPModel.create(voLBSP.size(), 1, CV_16UC1); //creation malloc row*times and then it will allocate the memory of the defined size


				for (auto i = voLBSP.begin(); i != voLBSP.end(); i++)
				{

					oDescriptorLBSPModel.at<ushort>(std::distance(voLBSP.begin(), i), 0) = *i;


				}
			
				oKAlgModel.calWeightedColor(oFramePutText3, oColor1, oLBSPROIModel);
			
				voBFTraining.push_back(oDescriptor1); // query descriptor

				bf.add(voBFTraining);
				bf.train();

			}



			if (iter1 != ((accumulator.begin() + (nStartFrame - 1)))){




				nFrame = nFrame + 1;

				nFrameCounter = nFrameCounter + 1;


				Rect trackROI;
				vector<Point2f> accum;
				oImage2 = imread(*iter1);


				oFramePutText2 = oImage2.clone();
				oFramePutText3 = oImage2.clone();
				oLBSPColor2 = oImage2.clone();
				oImageCopy2 = oImage2.clone();
				vector<KeyPoint> voDetectedKeys2;
				pSift->detect(oImage2, voDetectedKeys2);



				voKeyPoints2.resize(voDetectedKeys2.size());




				for (auto i = 0; i < voDetectedKeys2.size(); ++i){

					cv::Mat testDes = cv::Mat(1, 128, CV_32FC1);
					voKeyPoints2[i].key = voDetectedKeys2[i];
					voKeyPoints2[i].descriptor = Mat(1, 128, CV_32FC1);
					voKeyPoints2[i].index = -1;


				}



				voDetectedKeys2.clear();


				oKAlgNonModel.m_voKeyPoints = voKeyPoints2;
				std::ostringstream str;


				char frameString[10];
				char sym[2] = "#";
				itoa(nFrame, frameString, 10);
				strcat(frameString, sym);

				cv::putText(oFramePutText2, frameString, cv::Point(10, 20), FONT_HERSHEY_COMPLEX_SMALL, 1, CV_RGB(255, 0, 0));

				cv::putText(oFramePutText3, frameString, cv::Point(10, 20), FONT_HERSHEY_COMPLEX_SMALL, 1, CV_RGB(255, 0, 0));

				oDescriptor2 = oKAlgNonModel.computeDes(oImage2, oKAlgNonModel.m_voKeyPoints);



				for (auto i = 0; i < oDescriptor2.rows; ++i){
					{


						oKAlgNonModel.m_voKeyPoints[i].descriptor = oDescriptor2.row(i).clone();

					}
				}



				bf.radiusMatch(oDescriptor1, oDescriptor2, matchesR, nRadius); //descriptor2 train descriptor
			



#if DRAW == 13

				for (auto i = 0; i < oKAlgModel.m_voFilteredKeyPoints.size(); ++i){


					voDrawKeysModel.push_back(oKAlgModel.m_voFilteredKeyPoints[i].key);

				}



				for (auto i = 0; i < oKAlgNonModel.m_voKeyPoints.size(); ++i){


					voDrawKeysNonModel.push_back(oKAlgNonModel.m_voKeyPoints[i].key);

				}



				namedWindow("MATCHING", WINDOW_AUTOSIZE);
				drawMatches(oFramePutText1, voDrawKeysModel, oFramePutText2, voDrawKeysNonModel, matchesR, oMatches);
				

				
				imshow("MATCHING", oMatches);
				cvWaitKey(10);


				//voDrawKeysModel.clear();
				voDrawKeysNonModel.clear();
				
#endif

				for (unsigned int i = 0; i < oKAlgModel.m_voFilteredKeyPoints.size(); i++)
				{

					oKAlgModel.m_voFilteredKeyPoints[i].indi = 0;

				}








				oKAlgNonModel.filterMatches(matchesR);



				for (size_t k = 0; k < matchesR.size(); k++)

				{

					if (!matchesR[k].empty()){

						nTotalMatches += 1;
						cv::DMatch match = matchesR[k][0];
					



						if (match.queryIdx != -1 && match.queryIdx < oKAlgModel.m_voFilteredKeyPoints.size())

						if (match.trainIdx != -1 && match.trainIdx < oKAlgNonModel.m_voKeyPoints.size())


						{
							oKAlgModel.m_voFilteredKeyPoints.at(match.queryIdx).distance = match.distance;



							oKAlgModel.m_voFilteredKeyPoints.at(match.queryIdx).index = match.trainIdx;
							oKAlgModel.m_voFilteredKeyPoints.at(match.queryIdx).indi = 1;
							
							oKAlgNonModel.m_voKeyPoints[match.trainIdx].distance = match.distance;
						
							oKAlgNonModel.m_voKeyPoints[match.trainIdx].index = match.queryIdx;
							oKAlgNonModel.m_voKeyPoints[match.trainIdx].setAM = 1;
						

						}


					}

				}


				
	



				if (nFrame == (nStartFrame + 1))
				{

					previousCenter = center;

				}


				else
				{


					previousCenter = predCenter;

				}


				
				oKAlgModel.voting(oKAlgModel.m_voFilteredKeyPoints, voKeyPointsROI, oKAlgNonModel.m_voKeyPoints, accum, oFramePutText2, ROI, previousCenter, predCenter,
					nFrame, fExpo, fdisBCenter, oColor1, oColor2, fCompareHS);


				oKAlgModel.update(oKAlgModel.m_voFilteredKeyPoints, oKAlgNonModel.m_voKeyPoints, predCenter, fLearningRate, nFrame, fUpdate, fRateRemove, fCov, fSigma);


			

				if (nTotalMatches > 2)

				{


					oKAlgModel.getMinMaxWeightKeyPoint(oKAlgModel.m_voFilteredKeyPoints, oKAlgNonModel.m_voKeyPoints, voIndexes);



					oKAlgModel.computePairDistance(oKAlgModel.m_voFilteredKeyPoints, oKAlgNonModel.m_voKeyPoints, voIndexes, voDistance1, voDistance2);


					oKAlgModel.detectScaleChange(voDistance1, voDistance2, fScaleChange, fMeanScaleChange);
					


					if (fScaleChange >= 0.9 && fScaleChange < 1.1)
					{

						voAccScales.push_back(fScaleChange);
					}

					else{
					
					
						voAccScales.push_back(1.0);
					
					
					}
				}
				


				if (nFrameCounter == 11)
				{
					nFrameCounter = 0;
					

						
						fFinalScaleChange = std::accumulate(voAccScales.begin(), voAccScales.end(), 1.0, std::multiplies<float>());
					

					

					if (fFinalScaleChange > 0.9 && fFinalScaleChange < 1.1 )
					{

						

						ROI.width = fFinalScaleChange*ROI.width;
						ROI.height = fFinalScaleChange*ROI.height;

						trackROI.x = predCenter.x - (ROI.width / 2.0);

						trackROI.y = predCenter.y - (ROI.height / 2.0);

						trackROI.width = ROI.width;
						trackROI.height = ROI.height;

						voAccScales.clear();
						fFinalScaleChange = 1.0;
				
				}


					else
					{

						trackROI.x = predCenter.x - (ROI.width / 2.0);

						trackROI.y = predCenter.y - (ROI.height / 2.0);

						trackROI.width = ROI.width;
						trackROI.height = ROI.height;
					}

				}

				else
				{

					trackROI.x = predCenter.x - (ROI.width / 2.0);

					trackROI.y = predCenter.y - (ROI.height / 2.0);

					trackROI.width = ROI.width;
					trackROI.height = ROI.height;
				}



				trackROI.x = predCenter.x - (ROI.width / 2.0);

				trackROI.y = predCenter.y - (ROI.height / 2.0);

				trackROI.width = ROI.width;
				trackROI.height = ROI.height;


				oKAlgModel.ROIBoxAdjust(oImage2, trackROI);

				
				voIndexes.clear(), voDistance1.clear(), voDistance2.clear();
				nTotalMatches = 0;

# if SAVE == 2
				if (file1.isOpened())
				{

					file1 << "{" << "frame " << frame;
					file1 << "CX " << predCenter.x;
					file1 << "CY " << predCenter.y;
					file1 << "ROIW " << trackROI.width;
					file1 << "ROIH " << trackROI.height << "}";
				}

#endif		


# if SAVE == 3
				if (file1.isOpened())

				{

					file1 << "{" << "frame " << frame;
					file1 << "X " << trackROI.x;
					file1 << "Y " << trackROI.y;
					file1 << "W" << trackROI.width;
					file1 << "H " << trackROI.height << "}";
				}

#endif					

# if SAVE == 1
				if (file1.is_open())

				{


					file1 << trackROI.x << "," << trackROI.y << "," << trackROI.width << "," << trackROI.height << endl;
				}







				if (!file1.is_open())
				{
					if (nContext == 1)
					{

						file1.open(sResultDir + "\\" + sSeqName + "_TUNA_CONTEXT.txt ", ios::out);

					}


					if (nContext == 0)

					{

						file1.open(sResultDir + "\\" + sSeqName + "_TUNA_NO_CONTEXT.txt ", ios::out);

					}
				}

#endif					


			


				pt1 = Point2f(trackROI.x, trackROI.y);


				pt2 = Point2f((trackROI.x + trackROI.width), (trackROI.y + (trackROI.height)));

				oKAlgModel.ROIAdjust(oFramePutText2, pt1, pt2);

				rectangle(oFramePutText2, pt1, pt2, cv::Scalar(255, 0, 0), 2, 8);



				itoa(nFrame, frameString, 10);
				strcat(frameString, sym);

				cv::putText(oFramePutText2, frameString, cv::Point(10, 20), FONT_HERSHEY_COMPLEX_SMALL, 1, CV_RGB(255, 0, 0));





# if DRAW == 1
				namedWindow("OUTPUT", WINDOW_AUTOSIZE);

				imshow("OUTPUT", oFramePutText2);

				cvWaitKey(10);
# endif












				oKAlgNonModel.setLBSPROINonModel(oLBSPColor2, pt1, pt2, oLBSPROINonModel);



				if (oLBSPColor.size() != oLBSPColor2.size())
				{

					cv::resize(oLBSPColor2, oLBSPColor2, oLBSPColor.size());



				}


				cv::cvtColor(oLBSPColor2, oGrey2, CV_RGB2GRAY);
				ushort uLBSP2;  vector<ushort> voLBSP2;
				for (auto j = 2; j < (oGrey2.rows - 2); j++)
				{

					for (auto i = 2; i < (oGrey2.cols - 2); i++)

					{

						LBSP::computeGrayscaleDescriptor(oGrey2, oGrey.at<uchar>(j, i), i, j, THRESH, uLBSP2);

						voLBSP2.push_back(uLBSP2);

					}


				}




				oDescriptorLBSPNonModel.create(voLBSP2.size(), 1, CV_16UC1); //creation malloc row*times and then it will allocate the memory of the defined size


				for (auto i = voLBSP2.begin(); i != voLBSP2.end(); i++)
				{

					oDescriptorLBSPNonModel.at<ushort>(std::distance(voLBSP2.begin(), i), 0) = *i;

				}






				vector<int> voOutput;
				ObjectKeyPoint::matchLBSPDes(oDescriptorLBSPModel, oDescriptorLBSPNonModel, voOutput);

				int sum = std::accumulate(voOutput.begin(), voOutput.end(), 0);
				float fLBSPDesDiff = (float)sum / (16 * oGrey2.rows*oGrey2.cols);


				voOutput.clear();
				oDescriptorLBSPNonModel.empty(); // LBSPDesc Clear



				oKAlgModel.calWeightedColor(oFramePutText3, oColor2, oLBSPROIModel);

				fCompareHS = norm(oColor1, oColor2, NORM_L2);

			


				oHSModel.empty();    // HS Hist Clear
				oColor2.empty();
				oLBSPROINonModel.x = 0;
				oLBSPROINonModel.y = 0;
				oLBSPROINonModel.width = 0;
				oLBSPROINonModel.height = 0;

			
				if (oKAlgModel.m_voFilteredKeyPoints.size() > 500)
				{
					oKAlgModel.removeKeys(oKAlgModel.m_voFilteredKeyPoints, fThreshWeight, predCenter);

				}



	
				if ((fLBSPDesDiff >= 0.0 && fLBSPDesDiff <= 0.35) || (fCompareHS >= 0.0 && fCompareHS < 12.0))
			
				{



				
					
					oKAlgNonModel.nonModelKeysinTrackROI(oKAlgNonModel.m_voKeyPoints, oKAlgNonModel.m_voFilteredKeyPointsTrackROI, trackROI);






					oKAlgModel.addKeys(oKAlgNonModel.m_voFilteredKeyPointsTrackROI, oKAlgModel.m_voFilteredKeyPoints, voKeyPointsROI, trackROI, predCenter, fUpdate, fnewWeight);



					oKAlgModel.removeKeys(oKAlgModel.m_voFilteredKeyPoints, fThreshWeight, predCenter);




					oDescriptor1.empty();


					oDescriptor1.create(oKAlgModel.m_voFilteredKeyPoints.size(), oDescriptor1.cols, CV_32FC1);







					for (auto i = 0; i < oKAlgModel.m_voFilteredKeyPoints.size(); ++i)
					{


						oKAlgModel.m_voFilteredKeyPoints[i].descriptor.copyTo(oDescriptor1.row(i));



					}



					bf.clear();
					voBFTraining.clear();
					voBFTraining.push_back(oDescriptor1);
					bf.add(voBFTraining);
					bf.train();




				}




				oKAlgNonModel.m_voFilteredKeyPointsTrackROI.clear();

				oKAlgNonModel.setAMzero(oKAlgNonModel.m_voKeyPoints);
				


				oDescriptor2.empty();


				

				voKeyPoints2.clear();
				voKeyPointsROI.clear();
				matchesR.clear();
				
				
			


				for (auto i = 0; i < oKAlgModel.m_voFilteredKeyPoints.size(); ++i)
				{


					oKAlgModel.m_voFilteredKeyPoints[i].index = -1;



				}


				oFramePutText1.empty();
				oFramePutText1 = oImage2.clone();
				
				

			}

			
		} 


	
	} 

	
	file1.close();
	accumulator.clear();
	bf.clear();
	
	destroyWindow("OUTPUT");

#if SAVE == 2
	file1 << "]";
#endif
	return 0;


}

