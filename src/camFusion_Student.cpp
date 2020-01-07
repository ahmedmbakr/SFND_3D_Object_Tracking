
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

int countNumberOfMatchesBetweenTwoBoxes(const int currentFrameBB_id, const int prevFrameBB_id, const multimap <int, int> &multimapCurrentFrame,
    const multimap <int, int> &multimapPrevFrame);


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
 std::vector<cv::DMatch> &kptMatches)
{
    //cout << "Enter clusterKptMatchesWithROI\n";
    //If the distance is diff from the mean is less than this variable then it is considered as an error
    float maxAllowedDistance = 3;
    //Fill enclosed keypoints
    for(auto &kptCurr : kptsCurr)
    {
        if(boundingBox.roi.contains(kptCurr.pt))
            boundingBox.keypoints.push_back(kptCurr);
    }

    //Compute mean
    float mean = 0;
    int numMatchesInsideBB = 0;
    for(auto &kptMatch : kptMatches)
    {
        auto kptCurrIdx = kptMatch.trainIdx;
        auto kptPrevIdx = kptMatch.queryIdx;

        auto kptCurr = kptsCurr[kptCurrIdx];
        auto kptprev = kptsPrev[kptPrevIdx];

        auto absDist = cv::norm(kptCurr.pt - kptprev.pt);
        if(boundingBox.roi.contains(kptCurr.pt) && boundingBox.roi.contains(kptprev.pt))
        {
            mean += absDist;
            ++numMatchesInsideBB;
        }
    }
    mean /= numMatchesInsideBB;
    //cout<< "mean = " << mean << endl;
    for(auto &kptMatch : kptMatches)
    {
        auto kptCurrIdx = kptMatch.trainIdx;
        auto kptPrevIdx = kptMatch.queryIdx;

        auto kptCurr = kptsCurr[kptCurrIdx];
        auto kptprev = kptsPrev[kptPrevIdx];

        auto absDist = cv::norm(kptCurr.pt - kptprev.pt);
        if(boundingBox.roi.contains(kptCurr.pt) && boundingBox.roi.contains(kptprev.pt))
        {
            //cout<< "absDist = " << absDist<< endl;
            if(abs(absDist - mean) < maxAllowedDistance)
            {
                //cout << "Added new kptMatch to the bounding box\n";
                boundingBox.kptMatches.push_back(kptMatch);
            }
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    //TODO: implement visImg
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // compute camera-based TTC from distance ratios
    double meanDistRatio = std::accumulate(distRatios.begin(), distRatios.end(), 0.0) / distRatios.size();

    std::sort(distRatios.begin(), distRatios.end());
    double medianDistRatio = 0;
    if(distRatios.size() % 2 == 1)
        medianDistRatio =  distRatios[distRatios.size() / 2];
    else
    {
        medianDistRatio =  (distRatios[distRatios.size() / 2 - 1] + distRatios[distRatios.size() / 2]) / 2;
    }
    

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medianDistRatio);

}

bool compareTwoLidarPoints(LidarPoint& lp1, LidarPoint lp2)
{
    return lp1.x < lp2.x;
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // auxiliary variables
    float percentOfPointsToDecide = 0.1; //A number from 0 to 1 as example 0.1 means 10%
    int currNumberOfPointsToDecide = percentOfPointsToDecide * lidarPointsCurr.size();
    int prevNumberOfPointsToDecide = percentOfPointsToDecide * lidarPointsPrev.size();
    double dT = 1.0 / frameRate; // time between two measurements in seconds

    sort(lidarPointsPrev.begin(), lidarPointsPrev.end(), compareTwoLidarPoints);
    sort(lidarPointsCurr.begin(), lidarPointsCurr.end(), compareTwoLidarPoints);
    // find closest distance to Lidar points 
    double averageXPrev = 0;
    auto iteratorEnd = min(lidarPointsPrev.end(), lidarPointsPrev.begin() + prevNumberOfPointsToDecide);
    for(auto it=lidarPointsPrev.begin(); it!=iteratorEnd; ++it) {
        averageXPrev += it->x;
    }
    averageXPrev /= prevNumberOfPointsToDecide;

    double averageXCurr = 0;
    iteratorEnd = min(lidarPointsCurr.end(), lidarPointsCurr.begin() + currNumberOfPointsToDecide);
    for(auto it=lidarPointsCurr.begin(); it!=iteratorEnd; ++it) {
        averageXCurr += it->x;
    }
    averageXCurr /= currNumberOfPointsToDecide;

    // compute TTC from both measurements
    TTC = averageXCurr * dT / (averageXPrev-averageXCurr);
    cout << "TTC lidar = " << TTC <<endl;
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    cout << "Enter matching bounding boxes\n";
    multimap <int, int> multimapCurrentFrame;
    multimap <int, int> multimapPrevFrame;
    int matchesIdx = 0;
    for(auto matchesit = matches.begin(); matchesit != matches.end(); ++matchesit, ++matchesIdx)
    {
        int currentFrameKeypointIdx = matchesit->trainIdx;
        int prevFrameKeypointIdx = matchesit->queryIdx;

        auto currentFrameKeypoint = currFrame.keypoints[currentFrameKeypointIdx];
        auto prevFrameKeypoint = prevFrame.keypoints[prevFrameKeypointIdx];

        //find by which bounding boxes keypoints are enclosed in the previous and current frame.
        //Those are the potential candidates whose box ids I can store in a multimap
        for(auto& boundingBox : currFrame.boundingBoxes)
        {
            if(boundingBox.roi.contains(currentFrameKeypoint.pt))
            {
                multimapCurrentFrame.insert(pair<int, int>(boundingBox.boxID, matchesIdx));
            }
        }

        for(auto& boundingBox : prevFrame.boundingBoxes)
        {
            if(boundingBox.roi.contains(prevFrameKeypoint.pt))
            {
                multimapPrevFrame.insert(pair<int, int>(boundingBox.boxID, matchesIdx));
            }
        }
    }
    if(prevFrame.boundingBoxes.size() < currFrame.boundingBoxes.size())
    {
        for(auto& prevFrameBB : prevFrame.boundingBoxes)
        {
            int maxNumberMatchesFound = 0;
            int maxNumberMatchesFoundBoundedBoxIdInCurrFrame = -1;
            for(auto& currFrameBB : currFrame.boundingBoxes)
            {
                int countNumMatches = countNumberOfMatchesBetweenTwoBoxes(currFrameBB.boxID, prevFrameBB.boxID, multimapCurrentFrame, multimapPrevFrame);
                if(maxNumberMatchesFound < countNumMatches)
                {
                    maxNumberMatchesFound = countNumMatches;  
                    maxNumberMatchesFoundBoundedBoxIdInCurrFrame =  currFrameBB.boxID;       
                }
            }
            if(maxNumberMatchesFound > 0)
            {
                /*cout << "assigning "<< currFrameBB.boxID << " from current frame to " << 
                maxNumberMatchesFoundBoundedBoxIdInPrevFrame << " from prev frame and the number of matches points is "
                << maxNumberMatchesFound << "\n";*/
                bbBestMatches[prevFrameBB.boxID] = maxNumberMatchesFoundBoundedBoxIdInCurrFrame;
            }
        }
    }
    else
    {
        for(auto& currFrameBB : currFrame.boundingBoxes)
        {
            int maxNumberMatchesFound = 0;
            int maxNumberMatchesFoundBoundedBoxIdInPrevFrame = -1;
            for(auto& prevFrameBB : prevFrame.boundingBoxes)
            {
                int countNumMatches = countNumberOfMatchesBetweenTwoBoxes(currFrameBB.boxID, prevFrameBB.boxID, multimapCurrentFrame, multimapPrevFrame);
                if(maxNumberMatchesFound < countNumMatches)
                {
                    maxNumberMatchesFound = countNumMatches;  
                    maxNumberMatchesFoundBoundedBoxIdInPrevFrame =  prevFrameBB.boxID;       
                }
            }
            if(maxNumberMatchesFound > 0)
            {
                /*cout << "assigning "<< currFrameBB.boxID << " from current frame to " << 
                maxNumberMatchesFoundBoundedBoxIdInPrevFrame << " from prev frame and the number of matches points is "
                << maxNumberMatchesFound << "\n";*/
                bbBestMatches[maxNumberMatchesFoundBoundedBoxIdInPrevFrame] = currFrameBB.boxID;
            }
        }
    }
    
}

int countNumberOfMatchesBetweenTwoBoxes(const int currentFrameBB_id, const int prevFrameBB_id, const multimap <int, int> &multimapCurrentFrame,
    const multimap <int, int> &multimapPrevFrame)
{
    int countNumberOfMatches = 0;
    auto currentFramBB_iterSet = multimapCurrentFrame.equal_range(currentFrameBB_id);
    auto prevFramBB_iterSet = multimapPrevFrame.equal_range(prevFrameBB_id);
    
    for(auto currentIt = currentFramBB_iterSet.first; currentIt != currentFramBB_iterSet.second; ++currentIt)
    {
        int currFrameKeyPointMatcherIdx = currentIt->second;
        for(auto prevIt = prevFramBB_iterSet.first; prevIt != prevFramBB_iterSet.second; ++prevIt)
        {
            int prevFrameKeyPointMatcherIdx = prevIt->second;
            if(currFrameKeyPointMatcherIdx == prevFrameKeyPointMatcherIdx)
                ++countNumberOfMatches;
        }
    }
    return countNumberOfMatches;
}
