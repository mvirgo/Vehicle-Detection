from moviepy.editor import VideoFileClip
from IPython.display import HTML
from full_detection_pipeline import *
from lane_detect import *

def process_image(image):
    """ This processes through everything above.
    Will return the image with other vehicles shown boxed in blue,
    our own car position, lane curvature, and lane lines drawn.
    """
    # Vehicle Detection
    pre_result = find_cars(image)
    # Lane Detection (comment out to only have vehicle detection)
    result = draw_lines(pre_result)
    
    return result

# Convert to video
# vid_output is where the image will be saved to
vid_output = 'project_vid_output_with_lanes.mp4'

# The file referenced in clip1 is the original video before anything has been done to it
clip1 = VideoFileClip("project_video.mp4")

# NOTE: this function expects color images
vid_clip = clip1.fl_image(process_image) 
vid_clip.write_videofile(vid_output, audio=False)
