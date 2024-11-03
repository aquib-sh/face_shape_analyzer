# Face Shape Analyzer

A Python-based computer vision tool that analyzes facial features from photographs to determine face shape and provide personalized eyewear recommendations. Using OpenCV and dlib, this tool performs facial landmark detection to calculate key facial measurements and ratios, helping users understand their face shape characteristics.

## Features

- Face shape detection (Round, Oval, Square, Rectangle, Heart, Diamond)
- Facial measurements calculation
- Visual landmark plotting
- Eyewear style recommendations based on face shape
- Measurement ratio analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/face_shape_analyzer.git
cd face-shape-analyzer
```

2. Create and activate a virtual environment:
```bash
python3 -m venv myenv
source myenv/bin/activate  # On Windows, use: myenv\Scripts\activate
```

3. Install required packages:
```bash
pip install opencv-python dlib numpy
```

4. Download the shape predictor file:
```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

## Usage

```python
from face_shape_detector import get_face_shape

# Analyze an image
result = get_face_shape('path_to_your_photo.jpg')
print(result)
```

The output includes:
- Detected face shape
- Width-to-height ratio
- Jaw-to-cheekbone ratio
- Detailed facial measurements
- An annotated image showing facial landmarks

## Output Format

```python
{
    'face_shape': 'Round',
    'width_height_ratio': 1.34,
    'jaw_cheek_ratio': 0.7,
    'measurements': {
        'face_width': 597.01,
        'face_height': 445.03,
        'jaw_width': 399.02,
        'cheekbone_width': 573.01
    }
}
```

## Requirements

- Python 3.6+
- OpenCV
- dlib
- NumPy
- shape_predictor_68_face_landmarks.dat file

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- dlib for providing the facial landmark predictor
- OpenCV community for computer vision tools
