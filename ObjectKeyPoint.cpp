#include "opencv2/xfeatures2d/nonfree.hpp"

#include "ObjectKeyPoint.h"
#include <numeric>

using namespace std;

using namespace cv;
using std::max_element;
#define e 2.71828
#define pi 3.14
#define DRAW 13
# define SAVE 2


ObjectKeyPoint::ObjectKeyPoint()
{

}


ObjectKeyPoint::~ObjectKeyPoint()
{
}





cv::Mat ObjectKeyPoint::computeDes(const cv::Mat& oInitImg, std::vector<objectKeys>& m_voFilteredKeyPoints) {

	vector<KeyPoint> keys;

	for(auto i = 0; i < m_voFilteredKeyPoints.size(); ++i){

		keys.push_back(m_voFilteredKeyPoints[i].key);
	}

	Ptr<xfeatures2d::SIFT> pSift2 = xfeatures2d::SiftDescriptorExtractor::create();
	cv::Mat descriptor;
	pSift2->compute(oInitImg, keys, descriptor);
	m_voDescriptors = descriptor;
	return m_voDescriptors;
}






void ObjectKeyPoint::filteredKeyPoints(std::vector<objectKeys>& m_voKeyPoints, std::vector<objectKeys>& m_voFilteredKeyPoints, cv::Rect ROI, int contextFlag){





	for (std::vector<objectKeys>::iterator iter = m_voKeyPoints.begin(); iter != m_voKeyPoints.end(); ++iter)
	{

		

		if (contextFlag == 1)

		{
			if (iter->key.pt.x >= ROI.x && iter->key.pt.x <= (ROI.x + ROI.width) && iter->key.pt.y >= ROI.y && iter->key.pt.y <= ((1.2*ROI.y) + (ROI.height)))
			{
				
				m_voFilteredKeyPoints.push_back((*iter));
			}

		}


		else
		{


			if (iter->key.pt.x >= ROI.x && iter->key.pt.x <= (ROI.x + ROI.width) && iter->key.pt.y >= ROI.y && iter->key.pt.y <= ((ROI.y) + (ROI.height)))
			{
				
				m_voFilteredKeyPoints.push_back((*iter));


			}
		}


	}


}

void ObjectKeyPoint::filterBGKeyPoints(std::vector<objectKeys>& m_voKeyPoints, std::vector<objectKeys>& m_voBGKeyPoints, cv::Rect ROI, float fWeight)
{

	



	for (std::vector<objectKeys>::iterator iter = m_voKeyPoints.begin(); iter != m_voKeyPoints.end(); ++iter)
	{

		

			if (iter->key.pt.x >= ROI.x && iter->key.pt.x <= (ROI.x + ROI.width) && iter->key.pt.y >= ROI.y && iter->key.pt.y <= ((ROI.y) + (ROI.height)) )
			{
				iter->weight1 = fWeight;
				m_voBGKeyPoints.push_back(*iter);
			}

		}


		

	}

	

void ObjectKeyPoint::encodeStructure(std::vector<objectKeys>& m_voFilteredKeyPoints, cv::Point2f center, cv::Mat image1, float& fUpdate){
	

	for(unsigned int i = 0; i < m_voFilteredKeyPoints.size(); i++)
	{

		float x=0, m =0.01; float c= 0, y=0;
		Point2f dis_cen_temp =  center - m_voFilteredKeyPoints[i].key.pt;
		
		float dis_cen1 = (dis_cen_temp.x*dis_cen_temp.x) + (dis_cen_temp.y*dis_cen_temp.y);
	

		m_voFilteredKeyPoints[i].dis_Cen = dis_cen_temp; // L spatial constraint vector
		

		
		m_voFilteredKeyPoints[i].weight1 = std::max<float>((1 - abs(fUpdate*dis_cen1)),0.5);
		
	
		
#if DRAW == 1
		
		cv::arrowedLine(image1, cvPoint(m_voFilteredKeyPoints[i].key.pt.x, m_voFilteredKeyPoints[i].key.pt.y),  cvPoint(center.x, center.y), CV_RGB(0,0,255), 2 , 8);
		
		
		circle( image1,
			center,
			0,
			Scalar( 0, 255, 0 ),
			2,
			8 );
	
		namedWindow("APPEARANCE MODEL - PHASE 1", WINDOW_AUTOSIZE ); 
		moveWindow("APPEARANCE MODEL - PHASE 1", 200, 600);
		imshow("APPEARANCE MODEL - PHASE 1", image1);
		cvWaitKey(10);
		
		
#endif
		
	}

}





cv::Rect ObjectKeyPoint::readGT(string& gtFile)
{
size_t nStart = 0;
size_t nInit = 0;
size_t nTotal;
int nRect[4] = {};
ifstream groundFile;
groundFile.open(gtFile.c_str(),ios::in);
string sLine;
int nLineNumber = 1;
 if (groundFile.is_open())
  {
    while ( getline (groundFile,sLine) )
    {
		if(sLine.find(",") != string::npos)
		{
            // cout << sLine << " " << nLineNumber << endl;
			//cout << sLine << '\n';
		}
		 if (nLineNumber == 1)
			 break;
		
     }

      
    
   groundFile.close();
  }
 
  else
	  
  {cout << "Unable to open file"; 

}





nTotal = sLine.size();
for (auto i = 0; i < 4; i++)
{

nStart = sLine.find(",", nInit); //find from position start
string sX = sLine.substr(nInit, nStart); //from init pos till how many pos+count
nRect[i] = stoi(sX);
nInit = nStart+1;
}


Rect ROI(nRect[0], nRect[1], nRect[2], nRect[3]);
return ROI;
}



cv::Rect ObjectKeyPoint::readGTCenter(string& gtFile)
{
size_t nStart = 0;
size_t nInit = 0;
size_t nTotal;
int nRect[4] = {};
ifstream groundFile;
groundFile.open(gtFile.c_str(),ios::in);
string sLine;
int nLineNumber = 1;
 if (groundFile.is_open())
  {
    while ( getline (groundFile,sLine) )
    {
		if(sLine.find(",") != string::npos)
		{
             //cout << sLine << " " << nLineNumber << endl;
			//cout << sLine << '\n';
		}
		 if (nLineNumber == 1)
			 break;
		
     }

      
    
   groundFile.close();
  }
 
  else
	  
  {cout << "Unable to open file"; 

}





nTotal = sLine.size();
for (auto i = 0; i < 4; i++)
{

nStart = sLine.find(",", nInit); //find from position start
string sX = sLine.substr(nInit, nStart); //from init pos till how many pos+count
nRect[i] = stoi(sX);
nInit = nStart+1;
}


Rect ROI(nRect[0], nRect[1], nRect[2], nRect[3]);
return ROI;
}




void ObjectKeyPoint::splitFilename (const string& str)
{
  size_t found;
  size_t nTotal = str.size();
  cout << "Splitting: " << str << endl;
  found=str.find("img");
  cout << " folder: " << str.substr(0,found) << endl;
  
  
 
}




void ObjectKeyPoint::voting(std::vector<objectKeys>& m_voFilteredKeyPoints, std::vector<objectKeys>& keyPointsROI, std::vector<objectKeys>& m_voKeyPoints, std::vector<Point2f>& accum, cv::Mat image2, cv::Rect ROI, cv::Point2f& previousCenter, cv::Point2f& predCenter, int& frameNum, float& fExpo,
	float& distBetCenters, Mat oColor1, Mat oColor2, double& fCompareHS) {

	
	Point2f Xc, diffCenter;float diff; 
	
	
	// gaussian 5x5 pattern  based on fspecial('gaussian',[5 5], 6.0)

	float dataVal6[25] = { 0.0378  ,  0.0394 ,   0.0400 ,   0.0394 ,   0.0378,
		0.0394  ,  0.0411 ,   0.0417 ,   0.0411  ,  0.0394,
		0.0400  ,  0.0417 ,   0.0423 ,   0.0417  ,  0.0400,
		0.0394  ,  0.0411 ,   0.0417 ,   0.0411  ,  0.0394,
		0.0378 ,   0.0394 ,   0.0400 ,   0.0394  ,  0.0378 };

	cv::Mat gauss = cv::Mat(5, 5, CV_32FC1, dataVal6);

	Point2f res;
	cv::Mat ResX = Mat::zeros(image2.size(),CV_32FC1);
	cv::Mat ResY = Mat::ones(5,5,CV_32FC1);
	cv::Mat voteMatrix = Mat::zeros(image2.rows, image2.cols,CV_32FC1);
	//imshow("VOTE", voteMatrix);
	
	
	
	for(unsigned int i = 0; i < m_voFilteredKeyPoints.size(); i++)
	{
		
		
			if(m_voFilteredKeyPoints[i].indi == 1)
		{
			

			keyPointsROI.push_back(m_voKeyPoints[m_voFilteredKeyPoints[i].index]);

			m_voFilteredKeyPoints[i].nTimes = m_voFilteredKeyPoints[i].nTimes+1;
		
			if(frameNum == 2)

			{
				m_voFilteredKeyPoints[i].nTimes = 1;
				
			}

			else

			{
					m_voFilteredKeyPoints[i].nTimes = m_voFilteredKeyPoints[i].nTimes+1;
			}

		
			
			// VOTING BY ENCODED STRUCTURE
					Xc = m_voKeyPoints[m_voFilteredKeyPoints[i].index].key.pt + m_voFilteredKeyPoints[i].dis_Cen;
					
					
					m_voFilteredKeyPoints[i].predC = Xc; 
					m_voFilteredKeyPoints[i].predCenters.push_back(Xc);

					Point2f diff =  previousCenter - Xc; //PENALIZE MATCHING IF THE CURRENT PREDICTION IS TOO FAR FROM THE PREVIOUS CENTER

					

					int diff2 = (diff.x*diff.x + diff.y*diff.y);
					
					
					float expoDiff = exp(-diff2/fExpo); 
					
				
#if DRAW == 11
					cv::arrowedLine(image2, cvPoint(m_voKeyPoints[m_voFilteredKeyPoints[i].index].key.pt.x ,m_voKeyPoints[m_voFilteredKeyPoints[i].index].key.pt.y), cvPoint(Xc.x, Xc.y), Scalar(0,0,255), 1,8);
					
					circle( image2, cvPoint(Xc.x , Xc.y),  0,
						Scalar( 255, 0, 0 ),
						1,
						10 ); //Distance vector


					namedWindow("VOTING BY KEYPOINTS-PHASE 3", WINDOW_AUTOSIZE);
					moveWindow("VOTING BY KEYPOINTS-PHASE 3", 600, 400);

					imshow("VOTING BY KEYPOINTS-PHASE 3", image2);
					//imwrite(".\\Res\\voting\\VOTING BY KEYPOINTS-PHASE 3.jpg",voteMatrix);
					cvWaitKey(10);

#endif

#if SAVE == 2
					
					
				
# endif

					unsigned int p = int(Xc.x); //col
					unsigned int q = int(Xc.y); //rows
					
					unsigned int bound = 20; 


					cv::Point minLoc, maxLoc;
					double min, max;
					if (p >= bound  && p <= (image2.cols - bound) && q>=bound && q <= (image2.rows - bound) )
						
					{
						
						
						

						cv::Mat sub = voteMatrix(cv::Rect(p-gauss.cols/2,q- gauss.rows/2, gauss.cols, gauss.rows));
					
					

						float fValue = m_voFilteredKeyPoints[i].weight1*expoDiff;
						

						sub +=  m_voFilteredKeyPoints[i].weight1*gauss*expoDiff;
						
						
								
					
					}

					
					
					
			}

			else
			{
				
				if(frameNum == 2)
				{
					m_voFilteredKeyPoints[i].nTimes = 0;
				}
				else

				{m_voFilteredKeyPoints[i].nTimes = m_voFilteredKeyPoints[i].nTimes-1; //For how many frames the keypoint has not been detected
				}
				
			
			}
#if DRAW == 1

	namedWindow("VOTING BY KEYPOINTS IILUSTRATION IN VOTING MATRIX", WINDOW_AUTOSIZE ); 
	//moveWindow("VOTING BY KEYPOINTS IILUSTRATION IN VOTING MATRIX", 10, 50);
	imshow("VOTING BY KEYPOINTS IILUSTRATION IN VOTING MATRIX",voteMatrix);
	cvWaitKey(10);
	//imwrite(".\\Res\\voting\\voting.jpg",voteMatrix);
# endif			

	}

	

	
		
	

	float maximum1 = 0.0, maximum2 = 0.0; std::vector<Point2f> bestC1; std::vector<Point2f> bestC2;cv::Point2f zero(0, 0); int row=0, col =0;
	Point2f x;

	

	for(unsigned int j = 0 ; j < voteMatrix.rows; ++j)
	{
		
		for(unsigned int i = 0 ; i < voteMatrix.cols; ++i)
		{
				
			if(voteMatrix.at<float>(j,i) > (maximum1) && voteMatrix.at<float>(j,i) > 0.0 ){
				
				x.x = j;
				x.y = i;

				
	
				maximum1 = voteMatrix.at<float>(j,i);
	

				
				
			
			}
			
			
		}

	}
	

	if(maximum1 > 0.0 )
	{
		
			 int l = int(x.x);
			 int m = int(x.y);
				row = l;
				col = m;
				predCenter = Point2f(col, row);
				
				Point2f diff = predCenter - previousCenter;
			
				
	}

	else
	{
		
	
		if(frameNum == 2)
		{
			predCenter =  previousCenter;
		}

		else
		{		
		predCenter = predCenter;
		}
	}
		
	cv::Mat image3 = image2.clone();

					
	
}





void ObjectKeyPoint::update(std::vector<objectKeys>& m_voFilteredKeyPoints, std::vector<objectKeys>& m_voKeyPoints, cv::Point2f& predCenter, float& a,  int& frame, float& fUpdate, float& fRateRemove, float& fCov, float& fSigma){
	int nNumber = 0;
	for(unsigned int i = 0; i < m_voFilteredKeyPoints.size(); i++)
	{


				
		if ((m_voFilteredKeyPoints[i].indi == 1)){
			


			Point2f	diffCenter = m_voFilteredKeyPoints[i].predC - predCenter ;

					m_voFilteredKeyPoints[i].diffT1 = sqrt(diffCenter.x*diffCenter.x + diffCenter.y*diffCenter.y);
					
		

					m_voFilteredKeyPoints[i].proxFactor = std::max<float>((1 - abs(fUpdate*m_voFilteredKeyPoints[i].diffT1)),0.0);
				
					
				
					if( m_voFilteredKeyPoints[i].proxFactor == 0.0 )
			
			{
				
				
				m_voFilteredKeyPoints[i].weight1 =  m_voFilteredKeyPoints[i].weight1 - (fRateRemove)*m_voFilteredKeyPoints[i].weight1; 
			
			}

			else

			{

			
				m_voFilteredKeyPoints[i].weight1 = (1 - a)*m_voFilteredKeyPoints[i].weight1   + a*m_voFilteredKeyPoints[i].proxFactor; 
				
			
							
			}
	
		}

		else{ 
	

			m_voFilteredKeyPoints[i].weight1 =  m_voFilteredKeyPoints[i].weight1 - (fRateRemove)*m_voFilteredKeyPoints[i].weight1;
						
	

			
		}

		

	}
}



void::ObjectKeyPoint::nonModelKeysinTrackROI(std::vector<objectKeys>& oKeyPoints, std::vector<objectKeys>& oKeyPoints1, cv::Rect ROI)
{





	for (std::vector<objectKeys>::iterator iter = oKeyPoints.begin(); iter != oKeyPoints.end(); ++iter)
	{


		if (iter->key.pt.x >= ROI.x && iter->key.pt.x <= (ROI.x + ROI.width) && iter->key.pt.y >= ROI.y && iter->key.pt.y <= ((ROI.y) + (ROI.height)))
		{


			oKeyPoints1.push_back(*iter);
		}

	}

}













void ObjectKeyPoint::addKeys(std::vector<objectKeys>& m_voFilteredKeyPointsTrackROI, std::vector<objectKeys>& m_voFilteredKeyPoints, std::vector<objectKeys>& keyPointsROI, cv::Rect ROI, cv::Point2f& predCenter, float& fUpdate, float& fnewWeight){



	for (std::vector<objectKeys>::iterator iter = m_voFilteredKeyPointsTrackROI.begin(); iter != m_voFilteredKeyPointsTrackROI.end(); ++iter)


	{
		if (iter->setAM != 1 )
		{
			
			if (iter->predCenters.size() > 0)
			{

				cout << "here" << endl;
			}

			iter->dis_Cen = predCenter - iter->key.pt;




			Point2f dis_cen_temp = predCenter - iter->key.pt;

			float dis_cen1 = (dis_cen_temp.x*dis_cen_temp.x) + (dis_cen_temp.y*dis_cen_temp.y);

			iter->dis_Cen = dis_cen_temp; // X spatial constraint vector


			//iter->weight1 = fnewWeight;
			iter->weight1 = std::max<float>((1 - abs(0.002*dis_cen1)), 0.0);

			//cout << "ADD\t" << iter->weight1 << endl;

			m_voFilteredKeyPoints.push_back(*iter);
			

		}


		

	}




	


}


	

		
void ObjectKeyPoint::removeKeys(std::vector<objectKeys>& m_voFilteredKeyPoints, float& tWeight, cv::Point2f& predCenter){

		
	for(unsigned int i = 0; i < m_voFilteredKeyPoints.size(); i++)
	{
		
		{
			//cout << "WEIGHT \t" << m_voFilteredKeyPoints[i].weight1 << endl;
			if (m_voFilteredKeyPoints[i].weight1 <= tWeight /*|| m_voFilteredKeyPoints[i].nTimes > -20 || m_voFilteredKeyPoints[i].predCenters.size() == 0*/){

				
				
				//cout << "REMOVE \t" << m_voFilteredKeyPoints[i].weight1 << endl;
					m_voFilteredKeyPoints.erase(m_voFilteredKeyPoints.begin() + (i));
				
			}


		}

	}

	
}



void ObjectKeyPoint::setLBSPROIModel(cv::Mat& oImage, Point2f& pt1, Point2f& pt2, cv::Rect& oLBSPROI){

	if (pt1.x < 0)
	{
		pt1.x = 0;
	}

	if (pt1.x >= oImage.cols)
	{
		pt1.x = oImage.cols - 2;

	}

	if (pt2.y < 0)
	{
		pt2.y = 0;
	}

	if (pt2.y >= oImage.rows)
	{
		pt2.y = oImage.rows - 2;
	}
	
	int width = int(pt2.x - pt1.x);
	int height = int(pt2.y - pt1.y);
	oLBSPROI.x = int(pt1.x);
	oLBSPROI.y = int(pt1.y);
	oLBSPROI.width = width;
	oLBSPROI.height = height;
	oImage = oImage(oLBSPROI);
	
	

}



void ObjectKeyPoint::setLBSPROINonModel(cv::Mat& oImage, Point2f& pt1, Point2f& pt2, cv::Rect& oLBSPROI){

	if (pt1.x < 0)
	{
		pt1.x = 0;
	}

	if (pt1.x >= oImage.cols)
	{
		pt1.x = oImage.cols - 2;

	}

	if (pt2.y < 0)
	{
		pt2.y = 0;
	}

	if (pt2.y >= oImage.rows)
	{
		pt2.y = oImage.rows - 2;
	}

	int width = int(pt2.x - pt1.x);
	int height = int(pt2.y - pt1.y);
	oLBSPROI.x = int(pt1.x);
	oLBSPROI.y = int(pt1.y);
	oLBSPROI.width = width;
	oLBSPROI.height = height;
	oImage = oImage(oLBSPROI);
	

}



void ObjectKeyPoint::setAMzero(std::vector<objectKeys>& o_vKP)
{

	for (unsigned int i = 0; i < o_vKP.size(); i++)
	{

		o_vKP[i].setAM = 0;

	}



}








void ObjectKeyPoint::getMinMaxWeightKeyPoint(vector<objectKeys>& keyPoint1, vector<objectKeys>& keyPoint2, std::vector<int>& voIndexes)
{
	std::vector<float> voWeights, voWeightsNew; std::vector<int> voIndices; // make kps weight zero which have not been detected, keeping the index of keypoint
	
	
	for (auto i = 0; i < keyPoint1.size(); ++i)

	{
		if (keyPoint1[i].indi == 1)
		{

			voWeights.push_back(keyPoint1[i].weight1);
			voIndices.push_back(i);

		}

		

	}
	
	


	std::sort(voWeights.begin(), voWeights.end());
	
	


	size_t iHalfWeightSize = voWeights.size() / 2;

	size_t iWeightVectorIndex = (iHalfWeightSize + 1);



	for (auto i = iWeightVectorIndex; i < voWeights.size(); ++i)
	{

		voWeightsNew.push_back(voWeights[i]);
	}


	
	for (auto i = 0; i< voWeightsNew.size(); ++i)
		{

			for (auto j = 0; j < keyPoint1.size(); ++j)
			{



				if (voWeightsNew[i] == keyPoint1[j].weight1 && keyPoint1[j].indi == 1)
				{

					voIndexes.push_back(j);
				}

			}
	}




	int test2 = 1;


}



void ObjectKeyPoint::computePairDistance(std::vector<objectKeys>& keyPoint1, std::vector<objectKeys>& keyPoint2, std::vector<int>& voIndexes, std::vector<Point2f>& voDist1, std::vector<Point2f>& voDist2)
{
	Point2f pairedDist1, pairedDistNew1;
	
	size_t nSize = voIndexes.size();


	size_t i = voIndexes[nSize-1];
	

		for (auto k = 0; k < nSize; ++k)

		{


			pairedDist1 = keyPoint1[i].key.pt - keyPoint1[voIndexes[k]].key.pt;

			voDist1.push_back(pairedDist1);

			pairedDistNew1 = keyPoint2[keyPoint1[i].index].key.pt - keyPoint2[keyPoint1[voIndexes[k]].index].key.pt;

			voDist2.push_back(pairedDistNew1);
		}


		int test = 1;
	
}


void ObjectKeyPoint::detectScaleChange(std::vector<Point2f>& voDist1, std::vector<Point2f>& voDist2, float& scaleChange, float& scaleChange2)
{
	float fRatio;
	std::vector<float> voDisValues;



	for (auto i = 0; i < voDist1.size(); ++i)
	{

		fRatio = sqrt(voDist1[i].x*voDist1[i].x + voDist1[i].y*voDist1[i].y) / sqrt(voDist2[i].x*voDist2[i].x + voDist2[i].y*voDist2[i].y);

		if ((sqrt(voDist2[i].x*voDist2[i].x + voDist2[i].y*voDist2[i].y) != 0))
		{

			voDisValues.push_back(fRatio);
		}
	}

	

	size_t size = voDisValues.size();
	
	std::sort(voDisValues.begin(), voDisValues.end());

	

	if (voDisValues.size() > 2)
	{

		if (size % 2 == 0)
		{
			scaleChange = (voDisValues[size / 2 - 1] + voDisValues[size / 2]) / 2;
		}

		else
		{
			scaleChange = voDisValues[size / 2];

		}

		double sum = std::accumulate(voDisValues.begin(), voDisValues.end(), 0.0);
		scaleChange2 = sum / voDisValues.size();
	}


	
}











bool ObjectKeyPoint::sortDisCompare(const pair<KeyPoint,Point2f>& A, const pair<KeyPoint,Point2f>& B) {
	double X, Y;


	X = sqrt((A.second.x)*(A.second.x)+(A.second.y)*(A.second.y));
	Y = sqrt((B.second.x)*(B.second.x)+(B.second.y)*(B.second.y));


	return X < Y;


}





void ObjectKeyPoint::filterMatches(vector<vector<DMatch>>& foundMatches){


				for (size_t k = 0; k < foundMatches.size(); ++k)
				{

					if(foundMatches[k].size() > 1)
					{

						
						float ratio = foundMatches[k][0].distance/ foundMatches[k][1].distance;

						if(ratio > 0.9)
						{

							foundMatches[k].clear();
						}
						else
						{
						cv::DMatch match = foundMatches[k][0] ;

						foundMatches[k].clear();

						foundMatches[k].push_back(match);
					
					}
				}

}
}





void ObjectKeyPoint::calLBSPHist(vector<ushort>& gradientValue, cv::Mat& oDescriptor, cv::Mat& oLBSPHist, int& nKeys)
{
	int nBins = 256; //can reduce (in an interval)
	int nchannels[] = { 0 }; // index for LBSP channel
	int nHistSize = { nBins }; //no.of bins
	int nBinValue;
	int nGradValue;
	float fRange[] = { 0, 65536 }; //range of LBSP values as in how many max values can be one
	const float* fHistRange = { fRange };
	int nGradientRange = 65536;
	bool bUniform = true; bool bAccumulate = false;
	oLBSPHist = Mat::zeros(nBins,1, CV_16UC1);

	for (auto i = 0; i < gradientValue.size(); i++)
	{

		cout << gradientValue[i] << endl;
		nGradValue = (int)gradientValue[i];
		nBinValue = (nGradValue*nBins)/nGradientRange;
		oLBSPHist.at<ushort>(nBinValue, 0) += 1;

	}
	
}


void ObjectKeyPoint::ROIAdjust(const cv::Mat& oSource, Point2f & pt1, Point2f & pt2)
{

	if (pt2.x < 0)
	{
		pt2.x = 0;
	}

	if (pt2.x >= oSource.cols)
	{
		pt2.x = oSource.cols - 2;

	}

	if (pt2.y < 0)
	{
		pt2.y = 0;
	}

	if (pt2.y >= oSource.rows)
	{
		pt2.y = oSource.rows - 2;
	}

}



void ObjectKeyPoint::ROIBoxAdjust(const cv::Mat& oSource, cv::Rect& oBox)
{

	if (oBox.x < 0)
	{
		oBox.x = 0;
	}

	if (oBox.width >= oSource.cols)
	{
		oBox.width = oSource.cols -2;

	}

	if (oBox.y < 0)
	{
		oBox.y = 0;
	}

	if (oBox.height >= oSource.rows)
	{
		oBox.height = oSource.rows - 2;
	}

}

void ObjectKeyPoint::calWeightedColor(cv::Mat& oImage, cv::Mat& oHist, cv::Rect& oROI){


	Point oROICenterPixel;

	oROICenterPixel.x = (oROI.width) / 2;
	oROICenterPixel.y = (oROI.height) / 2;

	cv::Mat oTest = oImage.clone();
	oTest = oTest(oROI);





	float diagonal = sqrt(oROI.width*oROI.width + oROI.height*oROI.height);
	float sigma = diagonal / 6.0;

	float fGaussConst1 = 1.0 / (2 * pi*pow(sigma, 2));
	float fGaussConst2 = 2.0 * (pow(sigma, 2));
	float fGaussDist;
	const uchar* uImageData = oTest.data;
	const size_t stepRow = oTest.step.p[0];
	int nRedBinValue, nGreenBinValue, nBlueBinValue;

	int nRedBins = 16;
	int nGreenBins = 16;
	int nBlueBins = 16;
	int nRange = 256;
	
	//const int nSizes[3] = { nBlueBins, nGreenBins, nRedBins };

//	oHist = Mat::zeros(3, nSizes, CV_32FC1);
	
	Point Loc;

	for (auto i = 0; i < oTest.rows; i++)
	{

		for (auto j = 0; j < oTest.cols; j++)
		{

			Loc.x = j;
			Loc.y = i;
			float xLocCenter = (Loc.x - oROICenterPixel.x)*(Loc.x - oROICenterPixel.x);
			float yLocCenter = (Loc.y - oROICenterPixel.y)*(Loc.y - oROICenterPixel.y);
			float fGaussNum = (xLocCenter + yLocCenter) / fGaussConst2;
			float fExp = exp(-(fGaussNum));
			fGaussNum = fGaussConst1*fExp;
			const uchar* uPixelValueBlue = uImageData + stepRow*(Loc.y) + 3 * (Loc.x) + 0;
			const uchar* uPixelValueGreen = uImageData + stepRow*(Loc.y) + 3 * (Loc.x) + 1;
			const uchar* uPixelValueRed = uImageData + stepRow*(Loc.y) + 3 * (Loc.x) + 2;
			
			nBlueBinValue = ((int)*uPixelValueBlue*nBlueBins) / nRange;
			nGreenBinValue = ((int)*uPixelValueGreen*nGreenBins) / nRange;
			nRedBinValue = ((int)*uPixelValueRed*nRedBins) / nRange;
			
			oHist.at<float>(nBlueBinValue, nGreenBinValue, nRedBinValue) += fGaussNum;

		}
	}

	
}


void ObjectKeyPoint::calWeightedColorHist(cv::Mat& oImage, cv::Mat& oTest, Point2f& pt1, Point2f& pt2, cv::Mat& oHist, cv::Rect& oROI)

{
	Point oROICenterPixel;

	oROICenterPixel.x = (oROI.width)/2;
	oROICenterPixel.y = (oROI.height)/2;

	oTest = oImage.clone();
	oTest = oTest(oROI);

	
	cvtColor(oTest, oTest, CV_BGR2HSV);
	
	

	float diagonal = sqrt(oROI.width*oROI.width + oROI.height*oROI.height);
	float sigma = diagonal/6.0;

	float fGaussConst1 = 1.0/(2*pi*pow(sigma,2));
	float fGaussConst2 = 2.0 * (pow(sigma, 2));
	float fGaussDist;
	const uchar* uImageData = oTest.data;
	const size_t stepRow = oTest.step.p[0];
	int nHueBinValue, nSatBinValue;
	
	int nHueBins = 30;
	int nSatBins = 32;
	int nHueRange = 180;
	int nSatRange = 256;

	oHist = Mat::zeros(nHueBins, nSatBins, CV_32FC1);
	Point Loc;


	for (auto i = 0; i < oTest.rows; i++)
	{

		for (auto j = 0; j < oTest.cols; j++)
		{
			Loc.x = j;
			Loc.y = i;
			float xLocCenter = (Loc.x - oROICenterPixel.x)*(Loc.x - oROICenterPixel.x);
			float yLocCenter = (Loc.y - oROICenterPixel.y)*(Loc.y - oROICenterPixel.y);
			float fGaussNum = (xLocCenter + yLocCenter)/fGaussConst2;
			float fExp = exp(-(fGaussNum));
			fGaussNum = fGaussConst1*fExp;
			const uchar* uPixelValueHue = uImageData + stepRow*(Loc.y) + 3 * (Loc.x);
			const uchar* uPixelValueSat = uImageData + stepRow*(Loc.y) + 3 * (Loc.x) + 1;
			nHueBinValue = ((int)*uPixelValueHue*nHueBins)/nHueRange;
			nSatBinValue = ((int)*uPixelValueSat*nSatBins)/nSatRange;
			
			oHist.at<float>(nHueBinValue, nSatBinValue) += fGaussNum;
			
		}
	}




	
}


void ObjectKeyPoint::updateTemplateModel(cv::Mat& oModel1, cv::Mat& oImage1, cv::Mat& oModel2, cv::Mat& oImage2, cv::Mat& oHSModel, cv::Mat& oHSNonModel)
{



	oModel1.empty();
	
	oImage1.empty();
	oHSModel.empty();
	oHSModel = oHSNonModel.clone();

	oImage1 = oImage2.clone();

	oImage2.empty();

	oHSNonModel.empty();

	oModel1.create(oModel2.rows, 1, CV_16UC1);

	for (auto i = 0; i < oModel2.rows; i++)
	{
		oModel2.row(i).copyTo(oModel1.row(i));
		//cout << oDescriptorModel.at<ushort>(i, 0) << endl;
	}

	

	oModel2.empty();


}






















