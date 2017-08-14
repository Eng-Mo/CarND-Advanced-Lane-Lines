## Advanced Lane Detection Report
---
The project aims to investigate the classical computer vision techniques in order to detect Road Lane and measure the curvature under rough lighting condition.
### Project porcedure:
1. Compute the camera calibration matrix and distortion coefficients.
2. Apply distortion correction to each each frame.
3. Threshold lane lines using gradient and color threshold.
4. Extract region of the lane road.
5. Apply "Birds-eye view" perspective transform. 
6. Detect lane pixel and fit the lane.
7. Warp the fitted  lane to the original perspective.
8. Draw Lane on the original image.

### Project files
1. `LaneDetect.py` contains all required functions for Lane detections.
2. `Advanced-Lane-Detection.ipynb` contains the software python code pipeline.
---
[image1]: ./output_images/dis_undis.png "Calibration"
[image2]: ./output_images/test1.png 
[image3]: ./output_images/Lane_thresh.png
[image4]: ./output_images/roi-test1.png
[image5]: ./output_images/wraped-test1.png
[image6]: ./output_images/window-test1.png
[image7]: ./output_images/Lines-test1.png
[image8]: ./output_images/Lane-test1.png
[image9]: ./output_images/result.jpg

---

#### 1. Camera calibration.

The code for this step is contained in `def undistorT(imgorg)` function. I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![title][image1]

#### 2. Image segmentation
I used a combination of color and gradient thresholds to generate a binary image. I made threshold in two levels:
1. I computed the gray image gradient of the frame in respective x & y. Also, graduate magnitude was calculated and minimum threshold was applied as in the function `SobelThr(img)`. The ouput binary Image segmented the Lane clearly but under the lightness noise the the segmentation not very clear as following
2. I made color threshold for white and yellow color from `HSV,HLS and RGB` color spaces as shown in `ColorThreshold()`.
![title][image3]

#### 3. Region of interest
The function `region_of_interest()` extract a trapezoidal Lane road region from sigl chanell image based on the follwoing point `([[(200,675), (1200,675), (700,430),(500,430)]],dtype=np.int32)`.
![title][image4]


#### 3.Perspective transform.
The code for my perspective transform includes a function called `prespectI()`. The function takes as inputs an image (`img`), as well as the source (`src`) and destination (`dst`) points was chosen trial and error until produce the most parallel two lines.  I chose the hardcode the source and destination points in the following manner:

```python
 src=np.float32([[728,475],
                  [1058,690],
                  [242,690],
                  [565,475]])
    
 dst=np.float32([[1058,20],
                  [1058,700],
                  [242,700],
                  [242,20]])
```
This produced `warped ` image as the following image after applying `cv2.getPerspectiveTransform(src, dst)` that return Image matrix `M` and feed it to `cv2.warpPerspective()`.

![alt text][image5]

#### 4. Line fitting
`LineFitting()`idintifys the lane points and fit a second order polynomial to both right and left lane and the steps clearly commetted in `LaneDetect.py`. it started by computing the histogram of the half buttom of the image and extract 2 high peaks of x-position for right and left lane. After, 9 sliding windows to idintify lane pixels. Each one centered on the midpoint of the pixels from the window below. Extract the points of each lane lines using `polyfit()` that fits a 2nd order polynomial to each set of pixel:
![alt text][image6]
![alt text][image7]

#### 5. Curvature calculation
The curvature was calculated by choosing the maximum y-value corresponding to the bottom of the image `[y=719]`. Calculate the average curvature and then calculate the vehicle centre offset to the lane by assuming the midpoint image is the camera position. 

```python
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0]) 
    camera_center=wimgun.shape[0]/2
    car_position = (camera_center- (left_fitx[-1]+right_fitx[-1])/2)*xm_per_pix
```

#### 6. Final result.
![alt text][image8]
![alt text][image9]

---

### Pipeline (video)

Here's a [link to my video result](https://youtu.be/3PKT84lurqE) or download `project_output.mp4`

---

### Discussion

The proposed solution shows high accuracy in normal lighting condition and medium shadow level. However it fails on challenging video specially under the bridge. I think the classical method will not be a solution on every road as it impossible to tune the system to perform the same. the future work of this project is to use meanshift segmentation technique as it might help to remove the shadow. Also, A modern segmentation technique called semantic segmentation using deep learning might has high potential to segment the road at the implementation will be avaliable soon. 