import cv2
import dlib
import numpy as np
from math import atan2, degrees

def get_face_shape(image_path):
    """
    Analyze face shape from an image using facial landmarks
    Returns the predicted face shape and relevant measurements
    """
    # Initialize face detector and facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('tools/shape_predictor_68_face_landmarks.dat')
    
    # Read image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = detector(gray)
    if len(faces) == 0:
        return "No face detected"
    
    face = faces[0]
    landmarks = predictor(gray, face)
    points = np.zeros((68, 2), dtype="int")
    
    # Extract landmark coordinates
    for i in range(68):
        points[i] = (landmarks.part(i).x, landmarks.part(i).y)
    
    # Calculate measurements
    # Face width at temples (between points 0 and 16)
    face_width = np.linalg.norm(points[16] - points[0])
    
    # Face height (from chin to forehead)
    face_height = np.linalg.norm(points[8] - points[27])
    
    # Jaw width (between angles of jaw)
    jaw_width = np.linalg.norm(points[5] - points[11])
    
    # Cheekbone width (between most lateral points)
    cheekbone_width = np.linalg.norm(points[14] - points[2])
    
    # Calculate ratios
    width_height_ratio = face_width / face_height
    jaw_cheek_ratio = jaw_width / cheekbone_width
    
    # Determine face shape based on ratios
    if width_height_ratio > 1:
        if jaw_cheek_ratio > 0.9:
            face_shape = "Square"
        else:
            face_shape = "Round"
    else:
        if jaw_cheek_ratio > 0.9:
            face_shape = "Rectangle"
        elif jaw_cheek_ratio < 0.8:
            face_shape = "Heart"
        else:
            if cheekbone_width > face_width * 0.9:
                face_shape = "Diamond"
            else:
                face_shape = "Oval"
    
    # Draw landmarks and measurements on image
    for (x, y) in points:
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
    
    # Draw key measurements
    cv2.line(img, tuple(points[0]), tuple(points[16]), (255, 0, 0), 2)
    cv2.line(img, tuple(points[8]), tuple(points[27]), (255, 0, 0), 2)
    
    # Save annotated image
    output_path = 'face_analysis.jpg'
    cv2.imwrite(output_path, img)
    
    measurements = {
        'face_shape': face_shape,
        'width_height_ratio': round(width_height_ratio, 2),
        'jaw_cheek_ratio': round(jaw_cheek_ratio, 2),
        'measurements': {
            'face_width': round(face_width, 2),
            'face_height': round(face_height, 2),
            'jaw_width': round(jaw_width, 2),
            'cheekbone_width': round(cheekbone_width, 2)
        }
    }
    
    return measurements
