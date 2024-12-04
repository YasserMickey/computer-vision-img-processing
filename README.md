# Computer Vision Web Application - MSc Coursework

## Overview
A Flask-based web application that provides real-time image processing capabilities using OpenCV. This project was developed as part of a Master's degree in Data Science, demonstrating the practical implementation of various computer vision techniques.

## Features
- **Image Upload**: Support for uploading images through a web interface
- **Basic Image Processing**:
  - RGB Color Space Conversion
  - Grayscale Conversion
  - Gaussian Blur
- **Face Detection**:
  - Implementation using Haar Cascade Classifier
  - Real-time face detection with bounding box visualization
- **Edge Detection**:
  - Canny Edge Detection with multiple threshold options:
    - Wide threshold (30, 100)
    - Tight threshold (200, 240)
    - Optimized threshold detection


## Technologies Used
- **Backend**: Python, Flask
- **Computer Vision**: OpenCV (cv2)
- **Frontend**: HTML, Bootstrap
- **Image Processing**: NumPy, Matplotlib
- **Data Handling**: Base64 encoding for image transfer



## Project Structure
```
/
├── app.py                 # Main Flask application
├── templates/            
│   ├── active.html       # Main interface template
│   ├── index.html        # Upload page template
├── static/               # Static files (CSS, JS)
├── images/               # Uploaded images directory
└── requirements.txt      # Project dependencies
```

## Usage
1. Launch the application and navigate to the homepage
2. Upload an image using the provided interface
3. Use the navigation buttons to apply different image processing techniques:
   - View the original image
   - Apply basic processing (RGB, Grayscale, Blur)
   - Perform edge detection
   - Detect faces in the image

## Future Improvements
- Implementation of ROI & Color Selection
- Integration of K-Means Clustering for image segmentation
- Addition of U-Net Deep Learning capabilities
- Enhanced error handling and input validation
- Support for batch processing
- Real-time video processing capabilities


## Acknowledgments
- Developed as part of MSc Data Science coursework
- Dr. Atif Ahmad
- Middlesex University