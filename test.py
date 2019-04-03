import cv2
import scipy.misc
from skimage.io import imread
from skimage.color import rgb2gray
from dataloader import DataLoader
from harrisdetector import harris_corners
from pointTracker import PointTracker
from pointProjection import project_points
import numpy as np
import matplotlib.pyplot as plt
from debug.PointsVisualizer import PointVisualizer

dl = DataLoader('dataset/rgbd_dataset_freiburg2_desk') # Edit this string to load a different dataset

tracker = PointTracker()
vis = PointVisualizer()

# Set initial position of cameras in visualizer
initial_orientation, initial_position = dl.get_transform()
vis.set_groundtruth_transform(initial_orientation, initial_position)
vis.set_estimated_transform(initial_orientation, initial_position)

# Get points for the first frame
grey_img = dl.get_greyscale()
depth_img = dl.get_depth()


f = plt.figure()
f.add_subplot(2,2,1)
plt.imshow(grey_img)

f.add_subplot(2,2,2)
points_and_response = harris_corners(grey_img)
plt.imshow(points_and_response)

f.add_subplot(2,2,4)
grey_img = grey_img*255 #normalize for cornerharris method
grey_img = grey_img.astype(np.uint8) #convert to correct dtype

cv_harris = cv2.cornerHarris(grey_img,3,3,0.04)
cv_harris = cv_harris - np.amin(cv_harris)
cv_harris = cv_harris/np.amax(cv_harris)
cv_harris = cv_harris*255

print(np.amax(cv_harris))
print(np.amin(cv_harris))
plt.imshow(cv_harris)
plt.show()

tracker.add_new_corners(grey_img, points_and_response)

# Project the points in the first frame
previous_ids, previous_points = tracker.get_position_with_id()
previous_ids, previous_points = project_points(previous_ids, previous_points, depth_img)
vis.set_projected_points(previous_points, initial_orientation, initial_position)

current_orientation = initial_orientation
current_position = initial_position

while dl.has_next():
    dl.next()

    # Visualization
    gt_position, gt_orientation = dl.get_transform()
    vis.set_groundtruth_transform(gt_position, gt_orientation)

    # Get images
    grey_img = dl.get_greyscale()
    depth_img = dl.get_depth()
    
    # Track current points on new image
    #tracker.track_on_image(grey_img)
    #tracker.visualize(grey_img)

    # Project tracked points
    #ids, points = tracker.get_position_with_id()
    #ids, points = project_points(ids, points, depth_img)
    #vis.set_projected_points(points, gt_position, gt_orientation)

    # Replace lost points
    #points_and_response = harris_corners(grey_img)
    #tracker.add_new_corners(grey_img, points_and_response)

    # Find transformation of the new frame
    ## I will push this code to the repo a bit later, as there is still some smaller issues to sort out with it



cv2.destroyAllWindows()
