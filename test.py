from face_detect import *

raw_result = get_face_shape_and_recommendations("passport_size_photo.jpg")

# Format the results
formatted_result = format_measurements_for_humans(raw_result)
    
# Create readable output
readable_output = format_output_nicely(formatted_result)

print(readable_output)