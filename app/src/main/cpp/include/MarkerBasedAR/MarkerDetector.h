#pragma once

#include "Marker.h"

#define SPEEDMODE 1
#define ACCURATEMODE 2


class MarkerDetector
{
public:
    //Track
    std::vector<Marker> lastFrameMarkers;
    cv::Mat velocity;
    int frameCount = 0;
    std::vector<Marker> keyFrameMarkers;

    float lastMinMarkerArea = 2000;
    float minMarkerAreaAllowed = 2000;

    float m_minContourLengthAllowed;
    cv::Size markerSize;

    cv::Mat sourceImage;
    cv::Mat m_grayscaleImage;
    cv::Mat m_thresholdImg;
    cv::Mat canonicalMarkerImage;

    std::vector<Marker> markers;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Point2f> m_markerCorners2d;
    std::vector<cv::Point3f> m_markerCorners3d;
    cv::Mat camMatrix;
    cv::Mat distCoeff;
    int cannyThreshold;

    cv::Mat kernel;
    int mode = SPEEDMODE;

    MarkerDetector(float f, float dx, float dy, cv::Mat distortion) : markerSize(100, 100)
    {
        m_markerCorners2d.push_back(cv::Point2f(0, 0));
        m_markerCorners2d.push_back(cv::Point2f(markerSize.width - 1, 0));
        m_markerCorners2d.push_back(cv::Point2f(markerSize.width - 1, markerSize.height - 1));
        m_markerCorners2d.push_back(cv::Point2f(0, markerSize.height - 1));

        m_markerCorners3d.push_back(cv::Point3f(-0.5f, -0.5f, 0));
        m_markerCorners3d.push_back(cv::Point3f(+0.5f, -0.5f, 0));
        m_markerCorners3d.push_back(cv::Point3f(+0.5f, +0.5f, 0));
        m_markerCorners3d.push_back(cv::Point3f(-0.5f, +0.5f, 0));

        //((cv::Mat)cv::Mat::zeros(4, 1, CV_32F)).copyTo(distCoeff);
        distortion.copyTo(distCoeff);

        float m_intrinsic[3][3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                m_intrinsic[i][j] = 0;

        m_intrinsic[0][0] = f;
        m_intrinsic[1][1] = f;
        m_intrinsic[0][2] = dx;
        m_intrinsic[1][2] = dy;
        m_intrinsic[2][2] = 1;
        cv::Mat(3, 3, CV_32F, const_cast<float*>(&m_intrinsic[0][0])).copyTo(camMatrix);
        kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cannyThreshold = 400;
    }

    void setIntrinsic(float f, float dx, float dy)
    {
        camMatrix.at<float>(0, 0) = f;
        camMatrix.at<float>(1, 1) = f;
        camMatrix.at<float>(0, 2) = dx;
        camMatrix.at<float>(1, 2) = dy;
    }

    void setIntrinsic(float f, float dx, float dy, cv::Mat dist)
    {
        camMatrix.at<float>(0, 0) = f;
        camMatrix.at<float>(1, 1) = f;
        camMatrix.at<float>(0, 2) = dx;
        camMatrix.at<float>(1, 2) = dy;

        dist.copyTo(distCoeff);
    }

    int track(cv::Mat& sourceImg, std::vector<Marker>& detectedMarkers)
    {
        int ret = 0;
        frameCount++;
        if (frameCount % 10 == 0)
        {
            //findMarkers(sourceImg, detectedMarkers, false);
            getMarkersPos(sourceImg, detectedMarkers);
            ret = 1;
        }
        else {
            if (!velocity.empty())
            {
                //Mat img = Mat::zeros(sourceImg.size(), CV_8UC3);
                cv::Mat img(sourceImg.size(), CV_8UC3, cv::Scalar(255, 255, 255));
                bool markerMiss = false;
                for (int i = 0; i < lastFrameMarkers.size(); i++)
                {
                    cv::Mat twc = velocity * lastFrameMarkers[i].transformation.getMat();
                    //cout << velocity << endl;
                    //cout << twc << endl;
                    cv::Mat rotMat;
                    cv::Mat rvec, tvec;
                    twc.colRange(0, 3).rowRange(0, 3).copyTo(rotMat);
                    Rodrigues(rotMat, rvec);
                    //cout << "rvec : " << rvec << endl;
                    twc.rowRange(0, 3).col(3).copyTo(tvec);
                    //cout << rvec << endl << tvec << endl;
                    std::vector<cv::Point2f> ps;
                    projectPoints(m_markerCorners3d, rvec, tvec, camMatrix, distCoeff, ps);
                    cv::Rect r = boundingRect(ps);
                    //cout << r << endl;
                    int extendX = r.width * 0.5;
                    int extendY = r.height * 0.5;
                    cv::Rect rBig = cv::Rect(r.x - extendX, r.y - extendY, r.width + extendX * 2, r.height + extendY * 2);
                    if (!rectIn(sourceImg.size(), rBig))
                    {
                        markerMiss = true;
                        break;
                    }
                    else {
                        std::vector<Marker> partMarkers;
                        cv::Mat partImg;
                        sourceImg(rBig).copyTo(partImg);
                        cv::cvtColor(partImg, partImg, CV_RGB2GRAY);
                        /*imshow("part", partImg);
                        waitKey(0);*/
                        findMarkers(partImg, partMarkers, true);
                        for (int k = 0; k < partMarkers.size(); k++)
                        {
                            for (int j = 0; j < partMarkers[k].points.size(); j++)
                            {
                                partMarkers[k].points[j].x += rBig.x;
                                partMarkers[k].points[j].y += rBig.y;
                            }
                            bool repeat = false;
                            for (int j = 0; j < detectedMarkers.size(); j++)
                            {
                                if (detectedMarkers[j].id == partMarkers[k].id)
                                {
                                    repeat = true;
                                    break;
                                }
                            }
                            if (!repeat)
                            {
                                detectedMarkers.push_back(partMarkers[k]);
                            }

                        }
                    }
                    //cout << rBig << endl;

                    /*for (int j = 0; j < ps.size(); j++)
                    {
                        circle(img, ps[j], 2, Scalar(255, 0, 0));
                    }*/
                }

                if (markerMiss)
                {
                    velocity = cv::Mat();
                    lastFrameMarkers.clear();
                    //findMarkers(sourceImg, detectedMarkers);
                    getMarkersPos(sourceImg, detectedMarkers);
                    ret = 2;
                }
                else {
                    if (!detectedMarkers.size() == 0)
                    {
                        estimatePosition(detectedMarkers);
                        lastMinMarkerArea = 16 * minMarkerAreaAllowed;
                        for (int i = 0; i < detectedMarkers.size(); i++)
                        {
                            int area = boundingRect(detectedMarkers[i].points).area();
                            lastMinMarkerArea = area < lastMinMarkerArea ? area : lastMinMarkerArea;
                        }
                        ret = 3;
                    }
                    else
                    {
                        velocity = cv::Mat();
                        lastFrameMarkers.clear();
                        getMarkersPos(sourceImg, detectedMarkers);
                        //findMarkers(sourceImg, detectedMarkers);
                        ret = 4;
                    }

                    /*imshow("predict", img);
                    waitKey(0);*/
                }
            }
            else {
                ret = 5;
                getMarkersPos(sourceImg, detectedMarkers);
                //findMarkers(sourceImg, detectedMarkers);
            }
        }


        if (detectedMarkers.size() == 0)
        {
            velocity = cv::Mat();
            lastFrameMarkers.clear();
        }
        else
        {
            if (lastFrameMarkers.size() == 0)
            {
                velocity = cv::Mat();
            }
            else
            {
                velocity = cv::Mat();
                std::vector<cv::Mat> ves;
                for (int i = 0; i < lastFrameMarkers.size(); i++)
                {
                    for (int j = 0; j < detectedMarkers.size(); j++)
                    {
                        if (lastFrameMarkers[i].id == detectedMarkers[j].id)
                        {
                            cv::Mat twc1 = lastFrameMarkers[i].transformation.getMat();
                            cv::Mat twc2 = detectedMarkers[j].transformation.getMat();
                            //velocity = twc1.inv() * twc2;
                            velocity = twc2 * twc1.inv();
                            break;
                        }
                    }
                }
            }

            lastFrameMarkers = detectedMarkers;
        }

        return ret;
        //cout << detectedMarkers.size() << endl;
    }

    bool rectIn(cv::Size size, cv::Rect& rect)
    {
        int ltx = rect.x, lty = rect.y, rbx = rect.br().x, rby = rect.br().y;
        ltx = ltx >= 0 ? ltx : 0;
        lty = lty >= 0 ? lty : 0;
        rbx = rbx < size.width ? rbx : size.width - 1;
        rby = rby < size.height ? rby : size.height - 1;
        if (ltx >= size.width || lty >= size.height || rbx <= 0 || rby <= 0)
        {
            return false;
        }
        rect.x = ltx;
        rect.y = lty;
        rect.width = rbx - ltx + 1;
        rect.height = rby - lty + 1;
        if (!(0 <= rect.x && 0 <= rect.width && rect.x + rect.width <= size.width && 0 <= rect.y && 0 <= rect.height && rect.y + rect.height <= size.height))
        {
            return false;
        }
        return true;
    }

    bool getMarkersPos(cv::Mat& sourceImg, std::vector<Marker>& detectedMarkers)
    {
        sourceImg.copyTo(sourceImage);
        cv::Mat grayImg;
        cvtColor(sourceImage, grayImg, CV_RGB2GRAY);

        cv::Mat smallImg;
        //float scale = 0.25;
        float scale = (float)minMarkerAreaAllowed / lastMinMarkerArea;
        scale = sqrt(scale);
        scale = scale < 0.5 ? scale : 0.5;
        scale = scale > 0.25 ? scale : 0.25;
        //cout << "scale : " << scale << endl;
        //TimeLog::start();
        resize(grayImg, smallImg, cv::Size(), scale, scale);
        //TimeLog::end();

        std::vector<Marker> markers;
        //TimeLog::start();
        findMarkers(smallImg, markers, false);
        //cout << smallImg.size() << endl;
        //TimeLog::end();

        for (int i = 0; i < markers.size(); i++)
        {
            for (int j = 0; j < markers[i].points.size(); j++)
            {
                markers[i].points[j] /= scale;
            }
            cv::Mat partImg;
            cv::Rect r = boundingRect(markers[i].points);
            int extendX = r.width * 0.1;
            int extendY = r.height * 0.1;
            cv::Rect rBig = cv::Rect(r.x - extendX, r.y - extendY, r.width + extendX * 2, r.height + extendY * 2);
            if (rectIn(sourceImg.size(), rBig))
            {
                std::vector<Marker> partMarkers;
                grayImg(rBig).copyTo(partImg);
                findMarkers(partImg, partMarkers, true);
                for (int k = 0; k < partMarkers.size(); k++)
                {
                    for (int j = 0; j < partMarkers[k].points.size(); j++)
                    {
                        partMarkers[k].points[j].x += rBig.x;
                        partMarkers[k].points[j].y += rBig.y;
                    }
                    bool repeat = false;
                    for (int j = 0; j < detectedMarkers.size(); j++)
                    {
                        if (detectedMarkers[j].id == partMarkers[k].id)
                        {
                            repeat = true;
                            break;
                        }
                    }
                    if (!repeat)
                    {
                        detectedMarkers.push_back(partMarkers[k]);
                    }
                }
                //cout << partMarkers.size() << endl;
            }
        }
        /*if (keyFrameMarkers.size() == 0)
        {
            estimatePosition(detectedMarkers);
        }
        else
        {*/
        bool moved = false;
        estimatePosition(detectedMarkers);
        for (int i = 0; i < detectedMarkers.size(); i++)
        {
            Marker& m1 = detectedMarkers[i];
            bool exsit = false;
            for (int j = 0; j < keyFrameMarkers.size(); j++)
            {
                Marker& m2 = keyFrameMarkers[j];
                if (m1.id != m2.id) continue;
                exsit = true;
                float distSquared = 0;

                for (int c = 0; c < 4; c++)
                {
                    cv::Point v = m1.points[c] - m2.points[c];
                    distSquared += v.dot(v);
                }
                distSquared /= 4;
                if (distSquared < 20)
                {
                    m1.needEstimate = false;
                    m1.transformation = getMidTransform(m1.transformation, m2.transformation);
                    //m1.transformation = m2.transformation;
                }
                else
                {
                    moved = true;
                }
            }
            if (!exsit)
            {
                moved = true;
            }
        }
        //estimatePosition(detectedMarkers);
        if (moved)
        {
            keyFrameMarkers.clear();
            for (int i = 0; i < detectedMarkers.size(); i++)
            {
                keyFrameMarkers.push_back(detectedMarkers[i]);
            }
            //keyFrameMarkers = detectedMarkers;
        }
        //}


        if (detectedMarkers.size() == 0)
        {
            lastMinMarkerArea = 4 * minMarkerAreaAllowed;
        }
        else {
            lastMinMarkerArea = 16 * minMarkerAreaAllowed;
            for (int i = 0; i < detectedMarkers.size(); i++)
            {
                int area = boundingRect(detectedMarkers[i].points).area();
                lastMinMarkerArea = area < lastMinMarkerArea ? area : lastMinMarkerArea;
            }
        }
        //cout << "area : " << lastMinMarkerArea << endl;
        return true;
    }

    Transformation getMidTransform(Transformation& t1, Transformation& t2)
    {
        cv::Mat twc1 = t1.getMat();
        cv::Mat twc2 = t2.getMat();
        cv::Mat rotMat1, rotMat2;
        cv::Mat_<float> rotMat;

        cv::Mat rvec;

        twc1.colRange(0, 3).rowRange(0, 3).copyTo(rotMat1);
        twc2.colRange(0, 3).rowRange(0, 3).copyTo(rotMat2);

        rotMat = rotMat1.inv() * rotMat2;
        Rodrigues(rotMat, rvec);
        //cout << "rvec : " << rvec << endl;

        rvec = rvec / 2;
        //cout << "rvec : " << rvec << endl;
        Rodrigues(rvec, rotMat);
        rotMat = rotMat1 * rotMat;
        cv::Mat twc = cv::Mat::eye(4, 4, CV_32F);
        rotMat.copyTo(twc.colRange(0, 3).rowRange(0, 3));

        Transformation t;
        for (int col = 0; col < 3; col++)
        {
            for (int row = 0; row < 3; row++)
            {
                t.r().mat[row][col] = rotMat(row, col); // Copy rotation component
            }
            t.t().data[col] = (t1.t().data[col] + t2.t().data[col]) / 2;
        }
        return t;
    }

    void findMarkers(cv::Mat& img, std::vector<Marker>& markers, bool part)
    {
        cv::Mat thresholdImg;
        if (part)
        {
            cv::threshold(img, thresholdImg, 127, 255, cv::THRESH_BINARY_INV);
        }
        else {
            if (mode == SPEEDMODE)
            {
                cv::threshold(img, thresholdImg, 127, 255, cv::THRESH_BINARY_INV);
            }
            else {
                Canny(img, thresholdImg, 300, cannyThreshold);
                dilate(thresholdImg, thresholdImg, kernel);
                erode(thresholdImg, thresholdImg, kernel);
            }
            //imshow("thre", thresholdImg);
            /*imshow("thre", thresholdImg);
            waitKey(0);*/
        }

        std::vector<std::vector<cv::Point>> contours;
        myFindContours(thresholdImg, contours, 20);
        findCandidates(contours, markers);

        recognizeMarkers(img, markers);
    }

    bool getMarkerPos2(cv::Mat& sourceImg, std::vector<Marker>& detectedMarkers)
    {
        sourceImg.copyTo(sourceImage);
        cvtColor(sourceImg, m_grayscaleImage, CV_RGB2GRAY);

        cv::Mat tempImg;
        if (mode == SPEEDMODE)
        {
            resize(m_grayscaleImage, tempImg, cv::Size(0, 0), 0.5f, 0.5f);
        }
        else if (mode == ACCURATEMODE) {
            m_grayscaleImage.copyTo(tempImg);
        }
        //threshold(m_grayscaleImage, m_thresholdImg, 0, 255, THRESH_BINARY_INV | THRESH_TRIANGLE);

        //adaptiveThreshold(m_grayscaleImage, m_thresholdImg, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 0);

        Canny(tempImg, m_thresholdImg, 300, cannyThreshold);
        dilate(m_thresholdImg, m_thresholdImg, kernel);
        erode(m_thresholdImg, m_thresholdImg, kernel);
        myFindContours(m_thresholdImg, contours, 20);
        findCandidates(contours, detectedMarkers);
        if (mode == SPEEDMODE)
        {
            for (int i = 0; i < detectedMarkers.size(); i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    detectedMarkers[i].points[j] *= 2;
                }
            }
        }
        recognizeMarkers(m_grayscaleImage, detectedMarkers);
        estimatePosition(detectedMarkers);
        markers = detectedMarkers;
        return true;
    }

    void connectContours(const cv::Mat& imgIn, cv::Mat& imgOut)
    {
        imgIn.copyTo(imgOut);
        for (int r = 1; r < imgOut.rows - 1; r++)
        {
            for (int c = 1; c < imgOut.cols - 1; c++)
            {
                bool dilation = false;
                if (imgIn.at<uchar>(r, c) != 255)
                {
                    continue;
                }
                cv::Point p1, p2;
                int whiteNum = 0;
                for (int x = -1; x <= 1; x++)
                {
                    for (int y = -1; y <= 1; y++)
                    {
                        if (x == 0 && y == 0) continue;
                        if (imgIn.at<uchar>(r + x, c + y) == 255)
                        {
                            whiteNum++;
                            if (whiteNum == 1) p1 = cv::Point(x, y);
                            else if (whiteNum == 2) p2 = cv::Point(x, y);
                            else continue;
                        }
                    }
                }
                if (whiteNum == 1) dilation = true;
                else if (whiteNum == 2) {
                    if (p1.dot(p2) > 0)
                    {
                        dilation = true;
                    }
                }
                if (dilation)
                {
                    for (int x = -1; x <= 1; x++)
                    {
                        for (int y = -1; y <= 1; y++)
                        {
                            if (x == 0 && y == 0) continue;
                            imgOut.at<uchar>(r + x, y + c) = 255;
                        }
                    }
                }
            }
        }
    }

    void myFindContours(const cv::Mat& thresholdImg, std::vector<std::vector<cv::Point>>& contours, int minPointsAllowed)
    {
        std::vector<std::vector<cv::Point>> allContours;
        findContours(thresholdImg, allContours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
        contours.clear();
        for (int i = 0; i < allContours.size(); i++)
        {
            if (allContours[i].size() > minPointsAllowed)
            {
                contours.push_back(allContours[i]);
            }
        }
        /*Mat dstImg = Mat::zeros(thresholdImg.size(), CV_8UC3);
        drawContours(dstImg, contours, -1, Scalar(255, 255, 255), 1, 8);
        imshow("contours", dstImg);*/
    }

    void findCandidates
            (
                    const std::vector<std::vector<cv::Point>>& contours,
                    std::vector<Marker>& detectedMarkers
            )
    {
        std::vector<cv::Point>  approxCurve;
        std::vector<Marker>     possibleMarkers;

        // For each contour, analyze if it is a parallelepiped likely to be the marker
        for (size_t i = 0; i < contours.size(); i++)
        {
            // Approximate to a polygon
            double eps = contours[i].size() * 0.05;
            cv::approxPolyDP(contours[i], approxCurve, eps, true);

            // We interested only in polygons that contains only four points
            if (approxCurve.size() != 4)
                continue;

            // And they have to be convex
            if (!cv::isContourConvex(approxCurve))
                continue;

            // Ensure that the distance between consecutive points is large enough
            float minDist = std::numeric_limits<float>::max();

            for (int i = 0; i < 4; i++)
            {
                cv::Point side = approxCurve[i] - approxCurve[(i + 1) % 4];
                float squaredSideLength = side.dot(side);
                minDist = std::min(minDist, squaredSideLength);
            }

            // Check that distance is not very small
            if (minDist < 100)
                continue;

            // All tests are passed. Save marker candidate:
            Marker m;

            for (int i = 0; i < 4; i++)
                m.points.push_back(cv::Point2f(approxCurve[i].x, approxCurve[i].y));

            // Sort the points in anti-clockwise order
            // Trace a line between the first and second point.
            // If the third point is at the right side, then the points are anti-clockwise
            cv::Point v1 = m.points[1] - m.points[0];
            cv::Point v2 = m.points[2] - m.points[0];

            double o = (v1.x * v2.y) - (v1.y * v2.x);

            if (o < 0.0)		 //if the third point is in the left side, then sort in anti-clockwise order
                std::swap(m.points[1], m.points[3]);

            possibleMarkers.push_back(m);
        }



        // Remove these elements which corners are too close to each other.
        // First detect candidates for removal:
        std::vector< std::pair<int, int> > tooNearCandidates;
        for (size_t i = 0; i < possibleMarkers.size(); i++)
        {
            const Marker& m1 = possibleMarkers[i];

            //calculate the average distance of each corner to the nearest corner of the other marker candidate
            for (size_t j = i + 1; j < possibleMarkers.size(); j++)
            {
                const Marker& m2 = possibleMarkers[j];

                float distSquared = 0;

                for (int c = 0; c < 4; c++)
                {
                    cv::Point v = m1.points[c] - m2.points[c];
                    distSquared += v.dot(v);
                }

                distSquared /= 4;

                if (distSquared < 100)
                {
                    tooNearCandidates.push_back(std::pair<int, int>(i, j));
                }
            }
        }

        // Mark for removal the element of the pair with smaller perimeter
        std::vector<bool> removalMask(possibleMarkers.size(), false);

        for (size_t i = 0; i < tooNearCandidates.size(); i++)
        {
            float p1 = calPerimeter(possibleMarkers[tooNearCandidates[i].first].points);
            float p2 = calPerimeter(possibleMarkers[tooNearCandidates[i].second].points);

            size_t removalIndex;
            if (p1 > p2)
                removalIndex = tooNearCandidates[i].second;
            else
                removalIndex = tooNearCandidates[i].first;

            removalMask[removalIndex] = true;
        }

        // Return candidates
        detectedMarkers.clear();
        for (size_t i = 0; i < possibleMarkers.size(); i++)
        {
            if (!removalMask[i])
                detectedMarkers.push_back(possibleMarkers[i]);
        }
        /*Mat dstImg = Mat::zeros(sourceImage.size(), CV_8UC3);
        for (int i = 0; i < detectedMarkers.size(); i++)
        {
            detectedMarkers[i].draw(dstImg);
            imshow("possible", dstImg);
            waitKey(0);
        }*/
    }

    void recognizeMarkers(const cv::Mat& grayscale, std::vector<Marker>& detectedMarkers)
    {
        if (detectedMarkers.size() > 0)
        {
            std::vector<cv::Point2f> preciseCorners(4 * detectedMarkers.size());

            for (size_t i = 0; i < detectedMarkers.size(); i++)
            {
                const Marker& marker = detectedMarkers[i];

                for (int c = 0; c < 4; c++)
                {
                    preciseCorners[i * 4 + c] = marker.points[c];
                }
            }

            cv::TermCriteria termCriteria = cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 30, 0.01);
            cv::cornerSubPix(grayscale, preciseCorners, cvSize(5, 5), cvSize(-1, -1), termCriteria);

            // Copy refined corners position back to markers
            for (size_t i = 0; i < detectedMarkers.size(); i++)
            {
                Marker& marker = detectedMarkers[i];

                for (int c = 0; c < 4; c++)
                {
                    marker.points[c] = preciseCorners[i * 4 + c];
                }
            }
        }

        std::vector<Marker> goodMarkers;
        for (size_t i = 0; i < detectedMarkers.size(); i++)
        {
            Marker& marker = detectedMarkers[i];

            // Find the perspective transformation that brings current marker to rectangular form
            cv::Mat markerTransform = cv::getPerspectiveTransform(marker.points, m_markerCorners2d);

            // Transform image to get a canonical marker image
            cv::warpPerspective(grayscale, canonicalMarkerImage, markerTransform, markerSize);



            int nRotations;
            int id = Marker::getMarkerId(canonicalMarkerImage, nRotations);
            if (id != -1)
            {
                marker.id = id;
                //sort the points so that they are always in the same order no matter the camera orientation
                std::rotate(marker.points.begin(), marker.points.begin() + 4 - nRotations, marker.points.end());

                goodMarkers.push_back(marker);
            }
        }

        if (goodMarkers.size() > 0)
        {
            std::vector<cv::Point2f> preciseCorners(4 * goodMarkers.size());

            for (size_t i = 0; i < goodMarkers.size(); i++)
            {
                const Marker& marker = goodMarkers[i];

                for (int c = 0; c < 4; c++)
                {
                    preciseCorners[i * 4 + c] = marker.points[c];
                }
            }

            cv::TermCriteria termCriteria = cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 30, 0.01);
            cv::cornerSubPix(grayscale, preciseCorners, cvSize(7, 7), cvSize(-1, -1), termCriteria);

            // Copy refined corners position back to markers
            for (size_t i = 0; i < goodMarkers.size(); i++)
            {
                Marker& marker = goodMarkers[i];

                for (int c = 0; c < 4; c++)
                {
                    marker.points[c] = preciseCorners[i * 4 + c];
                }
            }
        }
        detectedMarkers = goodMarkers;
    }

    void estimatePosition(std::vector<Marker>& detectedMarkers)
    {

        for (int i = 0; i < detectedMarkers.size(); i++)
        {
            Marker& m = detectedMarkers[i];
            //if (!m.needEstimate) continue;

            cv::Mat Rvec;
            cv::Mat_<float> Tvec;
            cv::Mat raux, taux;
            cv::solvePnP(m_markerCorners3d, m.points, camMatrix, distCoeff, raux, taux);
            raux.convertTo(Rvec, CV_32F);
            taux.convertTo(Tvec, CV_32F);

            cv::Mat_<float> _Rvec;
            raux.convertTo(_Rvec, CV_32F);
            for (int i = 0; i < 3; i++)
            {
                m.rVec[i] = _Rvec(i);
            }

            cv::Mat_<float> rotMat(3, 3);
            cv::Rodrigues(Rvec, rotMat);

            // Copy to transformation matrix
            for (int col = 0; col < 3; col++)
            {
                for (int row = 0; row < 3; row++)
                {
                    m.transformation.r().mat[row][col] = rotMat(row, col); // Copy rotation component
                }
                m.transformation.t().data[col] = Tvec(col); // Copy translation component
            }

            // Since solvePnP finds camera location, w.r.t to marker pose, to get marker pose w.r.t to the camera we invert it.
            //m.transformation = m.transformation.getInverted();
        }
    }

    float calPerimeter(const std::vector<cv::Point2f> &a)
    {
        float sum = 0, dx, dy;

        for (size_t i = 0; i < a.size(); i++)
        {
            size_t i2 = (i + 1) % a.size();

            dx = a[i].x - a[i2].x;
            dy = a[i].y - a[i2].y;

            sum += std::sqrt(dx*dx + dy * dy);
        }

        return sum;
    }


};

