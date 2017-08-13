## Advanced Lane Detection Report
---
This project aims to detect Lane lines robustly during high lighting distortion. This report reprsents software pipline and preview two diffrent frames oitput at each stage. the first frame with normal lighting condition and second frame with high lighting disruption.

### Project files
1. `LaneDetect.py` contains all required functions for Lane detections.
2. `Advanced-Lane-Detection.ipynb` contanis the software python code pipeline.
---
[image1]: ./output_images/test1_r.png "distorted1"
[image2]: ./output_images/Undistored_test1_r.png "Udistorted1"
[image3]: ./output_images/test1.png "ROI"
[image4]: ./output_images/sobelTH-test1.png 
[image5]: ./output_images/colotTH-test1.png 
[image6]: ./output_images/combinedTH-test1.png
[image7]: ./output_images/roi-test1.png
[image8]: ./output_images/wraped-test1.png
[image9]: ./output_images/window-test1.png
[image10]: ./output_images/Lines-test1.png
[image11]: ./output_images/Lane-test1.png
[image12]: ./output_images/result.jpg



#### 1. Camera calibration.

The code for this step is contained in `def undistorT(imgorg)` function. I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![title][image1]
![title][image2]
#### 2. Image segmentation.
I used a combination of color and gradient thresholds to generate a binary image. I made thrshold in two levels:

1. I computed the gradiant of gray image of the frame in respective x & y. Also, graduate magnitude was calculated and minimum threshold was applied as in the function `SobelThr(img,n)`. The ouput binary Image segmented the Lane clearly in good lighting condition but with some noise in distorted frame with high lightness.
![title][image3]

Sobel gray threshold Frame1

![title][image4]

1. I made color threhsold for white and yellow color from diffrent color space as shown in `CTHR()` in `LaneDetect.py`

Color threshold

![title][image5]

Combined threshold

![title][image6]

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
This resulted `warped ` image as the following image after applying `cv2.getPerspectiveTransform(src, dst)` that return Image matrix `M` and feed it to `cv2.warpPerspective()`.

Warped image:
    
    ![alt text] [image8]

#### 4. Line fitting
`LineFitting()` in cell `[137]`idintifys the Lane pixels by computing the histogram of half buttom the image and extract 2 high peaks. After, 9 sliding windows for each lane pixels and extract the points of each lane lane lines with a 2nd order polynomial kinda like this:

```python
for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = wimgun.shape[0] - (window+1)*window_height
        win_y_high = wimgun.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, wimgun.shape[0]-1, wimgun.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
```

Window Fitting:
    
    ![alt text] [image9]
Line Fitting:
    ![alt text] [image10]
Line Fitting:
    ![alt text] [image11]

#### 5. Curvature calculation
The curvature was calculated by chosing the maximum y-value corresponding to the bottom of the image 
```python
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0]) 
```

#### 6. Final result.
The following are the results after unwarping the frames in two diffrent condditions. 
Result for test images:
    
    ![alt text] [image12]



---

### Pipeline (video)


Here's a [link to my video result](https://youtu.be/3PKT84lurqE) or download `project_output.mp4`

---

### Discussion

The proposed solution showed high accuracy in each frame. However the system is slow due to extracting Sobel edge. Also, it need to be tuned and other segmentation techniques such as watershed can produce robuster system. 