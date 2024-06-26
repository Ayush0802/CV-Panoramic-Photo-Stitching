# Panoramic Photo Stitching

## Description
This project is focused on creating panoramic images through an advanced image stitching process. Image stitching involves combining multiple overlapping images to create a single, cohesive panoramic image. This process is commonly used in photography, virtual tours, and mapping applications. This is a project which I created with my group members during my `Computer Vision` course. 

## Objectives
The main objectives of this project are:

- To explore and compare different feature extraction algorithms for image stitching.
- To implement an efficient stitching process with multiple blending methods to create smooth transitions.
- To provide a flexible and extensible framework for image stitching, allowing customization based on user requirements.

## Key Features
- Feature Extraction: The project supports a variety of feature extraction algorithms, including Harris, SIFT, BRISK, ORB, BRIEF, Shi-Tomasi, and FAST. This allows the user to choose the best algorithm based on the nature of the input images.
- Image Stitching: The stitching process involves finding the best shift between images using RANSAC, then aligning and blending the images to create a seamless panorama. - - Two blending methods are available: alpha blending and feather blending.
- End-to-End Alignment: The project includes a mechanism for end-to-end alignment, correcting y-axis shift errors for a more consistent panorama.
- Black Border Cropping: A cropping function removes black borders, ensuring the final stitched image is clean and visually appealing.
- Performance Metrics: The project provides metrics for the average number of matched features and total computation time for each feature extraction algorithm. This allows users to evaluate the efficiency and effectiveness of different approaches.
- Visualization: The project includes bar plots to visually compare the performance of different feature extraction algorithms.

## Run the main script
Go to the `src` folder of this project and run the below script :

    python ./main.py <input img dir>
    # for example
    python ./main.py ../input_image/hostel/


## Input format

The input dir should have:

- Some `.png` or `.jpg` images
- A `image_list.txt`, file should contain:
  - filename
  - focal_length

This is an example for `image_list.txt`:

```
# Filename   focal_length
hostel_01.jpg 240
hostel_02.jpg 240
hostel_03.jpg 240
hostel_04.jpg 240
hostel_05.jpg 240
```

## Outputs

Output of the resulted stitched image is stored in `result` folder for all the different feature extraction algorithms.

