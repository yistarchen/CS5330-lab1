import gradio as gr
import cv2
import numpy as np

def convert_color_space(image):
    converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return converted_image

def apply_threshold(image):
    lower_blue = np.array([100, 40, 40])
    upper_blue = np.array([140, 255, 255])
    thresholded_image = cv2.inRange(image, lower_blue, upper_blue)
    return thresholded_image

def morphological_operations(mask):
    kernel = np.ones((5,5), np.uint8)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    return opening

def identify_sky_pixels(image):
    hsv_image = convert_color_space(image)
    thresholded = apply_threshold(hsv_image)
    morphed = morphological_operations(thresholded)
    sky_visualization = cv2.bitwise_and(image, image, mask=morphed)
    return sky_visualization

def process_image(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    processed_image = identify_sky_pixels(image)
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    return processed_image

iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(),
    outputs=gr.Image(),
    title="Sky Pixel Identification",
    description="Upload an image to identify sky pixels."
)

iface.launch(share=True)