import json
import cv2
import dlib
import numpy as np
from math import atan2, degrees
from typing import Dict, Any

class EyewearRecommender:
    """
    Provides eyewear recommendations based on face shape and measurements
    """
    
    RECOMMENDATIONS = json.load(open("eyewear_help.json", "r"))
    
    @staticmethod
    def get_size_recommendations(measurements: Dict[str, float]) -> Dict[str, Any]:
        """Calculate ideal frame sizes based on facial measurements"""
        face_width = measurements['face_width']
        face_height = measurements['face_height']
        
        return {
            "ideal_frame_width": round(face_width * 0.9, 2),  # Slightly narrower than face width
            "ideal_lens_height": round(face_height * 0.25, 2),  # About 1/4 of face height
            "bridge_width": round(face_width * 0.1, 2),  # Approximately 1/10 of face width
            "temple_length": round(face_width * 1.1, 2)  # Slightly longer than face width
        }

def get_face_shape_and_recommendations(image_path: str) -> Dict[str, Any]:
    """
    Analyze face shape and provide eyewear recommendations
    Returns both face measurements and specific eyewear suggestions
    """
    # Original face detection code remains the same
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('tools/shape_predictor_68_face_landmarks.dat')
    
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)
    if len(faces) == 0:
        return {"error": "No face detected"}
    
    face = faces[0]
    landmarks = predictor(gray, face)
    points = np.zeros((68, 2), dtype="int")
    
    for i in range(68):
        points[i] = (landmarks.part(i).x, landmarks.part(i).y)
    
    # Calculate measurements
    face_width = np.linalg.norm(points[16] - points[0])
    face_height = np.linalg.norm(points[8] - points[27])
    jaw_width = np.linalg.norm(points[5] - points[11])
    cheekbone_width = np.linalg.norm(points[14] - points[2])
    
    # Calculate ratios
    width_height_ratio = face_width / face_height
    jaw_cheek_ratio = jaw_width / cheekbone_width
    
    # Determine face shape (original logic)
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
    
    # Create measurements dictionary
    measurements = {
        'face_width': face_width,
        'face_height': face_height,
        'jaw_width': jaw_width,
        'cheekbone_width': cheekbone_width
    }
    
    # Get eyewear recommendations
    recommender = EyewearRecommender()
    frame_sizes = recommender.get_size_recommendations(measurements)
    
    # Combine all results
    result = {
        'face_shape': face_shape,
        'width_height_ratio': round(width_height_ratio, 2),
        'jaw_cheek_ratio': round(jaw_cheek_ratio, 2),
        'measurements': {
            'face_width': round(face_width, 2),
            'face_height': round(face_height, 2),
            'jaw_width': round(jaw_width, 2),
            'cheekbone_width': round(cheekbone_width, 2)
        },
        'eyewear_recommendations': {
            'general': EyewearRecommender.RECOMMENDATIONS[face_shape],
            'frame_sizes': frame_sizes
        }
    }
    
    # Draw landmarks and measurements on image
    for (x, y) in points:
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
    
    cv2.line(img, tuple(points[0]), tuple(points[16]), (255, 0, 0), 2)
    cv2.line(img, tuple(points[8]), tuple(points[27]), (255, 0, 0), 2)
    
    output_path = 'face_analysis.jpg'
    cv2.imwrite(output_path, img)
    
    return result


def format_measurements_for_humans(result: dict) -> dict:
    """
    Convert numpy float values to human-readable measurements in millimeters
    and round decimals to 2 places for better readability.
    
    Args:
        result (dict): Original output dictionary with numpy float values
        
    Returns:
        dict: Formatted dictionary with human-readable values
    """
    # Deep copy to avoid modifying original
    formatted = result.copy()
    
    # Convert pixel measurements to millimeters (assuming 96 DPI)
    pixel_to_mm = 0.26458333  # 1 pixel = 0.26458333 mm at 96 DPI
    
    # Format frame sizes
    frame_sizes = formatted['eyewear_recommendations']['frame_sizes']
    formatted['eyewear_recommendations']['frame_sizes'] = {
        'bridge_width': f"{float(frame_sizes['bridge_width'] * pixel_to_mm):.1f}mm",
        'ideal_frame_width': f"{float(frame_sizes['ideal_frame_width'] * pixel_to_mm):.1f}mm",
        'ideal_lens_height': f"{float(frame_sizes['ideal_lens_height'] * pixel_to_mm):.1f}mm",
        'temple_length': f"{float(frame_sizes['temple_length'] * pixel_to_mm):.1f}mm"
    }
    
    # Format face measurements
    measurements = formatted['measurements']
    formatted['measurements'] = {
        'cheekbone_width': f"{float(measurements['cheekbone_width'] * pixel_to_mm):.1f}mm",
        'face_height': f"{float(measurements['face_height'] * pixel_to_mm):.1f}mm",
        'face_width': f"{float(measurements['face_width'] * pixel_to_mm):.1f}mm",
        'jaw_width': f"{float(measurements['jaw_width'] * pixel_to_mm):.1f}mm"
    }
    
    # Format ratios
    formatted['width_height_ratio'] = f"{float(formatted['width_height_ratio']):.2f}"
    formatted['jaw_cheek_ratio'] = f"{float(formatted['jaw_cheek_ratio']):.2f}"
    
    return formatted

def format_output_nicely(result: dict) -> str:
    """
    Create a nicely formatted string output of the face analysis results.
    
    Args:
        result (dict): Formatted dictionary with human-readable values
        
    Returns:
        str: Nicely formatted string output
    """
    output = []
    output.append("🔍 FACE SHAPE ANALYSIS")
    output.append("=" * 50)
    output.append(f"Face Shape: {result['face_shape']}")
    output.append(f"Width-to-Height Ratio: {result['width_height_ratio']}")
    output.append(f"Jaw-to-Cheek Ratio: {result['jaw_cheek_ratio']}")
    
    output.append("\n📏 MEASUREMENTS")
    output.append("-" * 50)
    for key, value in result['measurements'].items():
        output.append(f"{key.replace('_', ' ').title()}: {value}")
    
    output.append("\n👓 EYEWEAR RECOMMENDATIONS")
    output.append("-" * 50)
    output.append("\nIdeal Frame Measurements:")
    for key, value in result['eyewear_recommendations']['frame_sizes'].items():
        output.append(f"{key.replace('_', ' ').title()}: {value}")
    
    output.append("\nRecommended Styles:")
    for style in result['eyewear_recommendations']['general']['best_styles']:
        output.append(f"✓ {style}")
    
    output.append("\nStyles to Avoid:")
    for style in result['eyewear_recommendations']['general']['avoid']:
        output.append(f"✗ {style}")
    
    output.append("\nKey Features to Look For:")
    for feature in result['eyewear_recommendations']['general']['features_to_look_for']:
        output.append(f"• {feature}")
    
    output.append(f"\nWhy These Recommendations: {result['eyewear_recommendations']['general']['why']}")
    
    return "\n".join(output)