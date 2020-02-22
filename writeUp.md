# Camera Based 3D Object Tracking

## FP.0 Final Report

This document represents a writup document for the project that explains each of the rubric points.

## FP.1 Match 3D Objects

Implement the method `matchBoundingBoxes` starting from line 293 in camFusion_Student.cpp file, which takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property).
Matches are the ones with the highest number of keypoint correspondences.

## FP.2 Compute Lidar-based TTC

Compute the time-to-collision in second for all matched 3D objects using only Lidar measurements from the matched bounding boxes between current and previous frame is done in the method `computeTTCLidar` starting from line 259 in camFusion_Student.cpp file.

## FP.3 Associate Keypoint Correspondences with Bounding Boxes

Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition must be added to a vector in the respective bounding box which is done in the method `clusterKptMatchesWithROI` starting from line 137 in camFusion_stadent.cpp file.

## FP.4 Compute Camera-based TTC

Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame which is done in `computeTTCCamera` function starting from line 193 in camFusion_student.cpp.

## FP.5, FP.6 Performance Evaluation 1, 2

I implemented a function called `performanceEvaluationMain` inside FinalProject_Camera.cpp file starting from line 317 that is responsible for running all possible combinations of detectors, descriptors, matchers, matcher descriptors, and matcher selectors for all images. It includes six nested loops to find all the possible combinations and produces the result as a CSV file.
The CSV file is converted manually to excel sheet for analysis and graph drawing.
The excel sheet can be found in the root directory with the name `SFND_3D_Object_Tracking.xlsx`.
The output can be shown in the first `output` tab and the graph can be shown in the second `Graph` tab in the excel sheet.

As seen in the excel sheet the following cases show extreme Lidar reading:

- For camera image 3, the Lidar reading is far away from the correct TTC as the output of the camera is around 12m but the Lidar output is 20m

- For camera image 4, the Lidar reading is far away from the correct TTC as the output of the camera is around 12m but the Lidar output is 14m

- For camera image 8, the Lidar reading is far away from the correct TTC as the output of the camera is around 13.7m but the Lidar output is 15m

My reason is that the used coco deep neural network can not accurately identify the car in the previously discussed images and it leads to fewer number of points to be matched inside the car boxes and fair point matching that results in wrong TTC estimations

As seen in the excel sheet the following cases shows  extreme camera readings:

- In line 131 the combination: [HARRIS, BRISK,MAT_BF,DES_BINARY, SEL_KNN, 0000000005.png] results in not a number reading

- In line 131 the combination: [HARRIS, BRISK,MAT_BF,DES_BINARY, SEL_KNN, 0000000009.png] results in not a number reading

- In line 131 the combination: [HARRIS, BRISK,MAT_BF,DES_HOG, SEL_KNN, 0000000005.png] results in not a number reading

My obvious reason is that the combination for HARRIS, BRISK, MAT_BF, SEL_KNN is not a good match.
