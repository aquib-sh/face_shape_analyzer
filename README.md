# Face Shape Analyzer with Eyewear Recommendations

A Python-based computer vision tool that analyzes facial features from photographs to determine face shape and provide detailed eyewear recommendations. Using OpenCV and dlib, this tool performs facial landmark detection to calculate key facial measurements and ratios, helping users understand their face shape characteristics and find the most suitable eyewear styles.

## Features

- Face shape detection (Round, Oval, Square, Rectangle, Heart, Diamond)
- Precise facial measurements and ratios
- Ideal frame size calculations
- Comprehensive eyewear style recommendations
- Visual landmark plotting
- Human-readable measurement outputs
- Detailed style guides for each face shape

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
mkdir tools
mv shape_predictor_68_face_landmarks.dat tools/shape_predictor_68_face_landmarks.dat
```

## Usage

```python
# Basic face shape analysis
result = get_face_shape_and_recommendations('path_to_your_photo.jpg')

# Get human-readable output
formatted_data, readable_output = analyze_face_with_readable_output('path_to_your_photo.jpg')
print(readable_output)
```

## Output Format

The tool provides both raw measurements and human-readable output:

### Raw Measurements
```python
{
    'face_shape': 'Round',
    'width_height_ratio': '1.34',
    'jaw_cheek_ratio': '0.70',
    'measurements': {
        'face_width': '158.0mm',
        'face_height': '117.7mm',
        'jaw_width': '105.6mm',
        'cheekbone_width': '151.6mm'
    }
}
```

### Eyewear Recommendations
```python
{
    'frame_sizes': {
        'bridge_width': '15.8mm',
        'ideal_frame_width': '142.2mm',
        'ideal_lens_height': '29.4mm',
        'temple_length': '173.8mm'
    },
    'general': {
        'best_styles': ['Rectangle/Square frames', ...],
        'avoid': ['Round frames', ...],
        'features_to_look_for': ['Strong bridge', ...],
        'why': 'Explanation of recommendations'
    }
}
```

## Features in Detail

### Face Shape Detection
- Analyzes facial landmarks to determine face shape
- Calculates key ratios and proportions
- Provides visual representation of measurements

### Measurements
- Face width and height
- Jaw width
- Cheekbone width
- Width-to-height ratio
- Jaw-to-cheek ratio

### Eyewear Recommendations
- Ideal frame measurements
- Best frame styles for face shape
- Styles to avoid
- Key features to look for
- Explanation of recommendations

## Requirements

- Python 3.6+
- OpenCV
- dlib
- NumPy
- shape_predictor_68_face_landmarks.dat file

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas for improvement include:

- Additional face shapes and measurement points
- Enhanced recommendation algorithms
- UI/UX improvements
- Performance optimizations
- Additional output formats

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- dlib for providing the facial landmark predictor
- OpenCV community for computer vision tools
- Thanks to all contributors and testers

## Note

This tool is designed for reference purposes only. For the best eyewear fit, we recommend consulting with an eyewear professional and trying frames on in person.