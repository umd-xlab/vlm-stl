import os
import numpy as np
import cv2

# ROS
from sensor_msgs.msg import Image

# img
from PIL import Image as PILImage


def msg_to_pil(msg: Image) -> PILImage.Image:
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, -1)
    pil_image = PILImage.fromarray(img)
    return pil_image

def yuvy_to_pil(image_msg: Image) -> PILImage.Image:
    try:
        # Extract the raw image data from the ROS message
        width = image_msg.width
        height = image_msg.height
        yuyv_data = np.frombuffer(image_msg.data, dtype=np.uint8)

        # Ensure the data matches YUYV format (2 bytes per pixel pair)
        yuyv_image = yuyv_data.reshape((height, width, 2))

        # Convert YUYV to RGB using OpenCV
        rgb_image = cv2.cvtColor(yuyv_image, cv2.COLOR_YUV2RGB_YUYV)

        # Convert the RGB image (numpy array) to a PIL Image
        pil_image = PILImage.fromarray(rgb_image)

        return pil_image

    except Exception as e:
        print(f"Failed to convert image message to PIL: {e}")
        return None


def pil_to_msg(pil_img: PILImage.Image, encoding="mono8") -> Image:
    img = np.asarray(pil_img)
    ros_image = Image(encoding=encoding)
    ros_image.height, ros_image.width, _ = img.shape
    ros_image.data = img.ravel().tobytes()
    ros_image.step = ros_image.width
    return ros_image

