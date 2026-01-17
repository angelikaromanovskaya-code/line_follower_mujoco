import cv2
import numpy as np

class LineFollowerCV:
    
    def __init__(self, img_width = 64, img_height = 64):
        self.roi_size = 8
        self.img_height = img_height
        self.img_width = img_width

        self.roi_left = (self.img_width // 4 - self.roi_size // 2, self.img_height // 2 - self.roi_size // 2)
        self.roi_center = (self.img_width // 2 - self.roi_size // 2, self.img_height // 2 - self.roi_size // 2)
        self.roi_right = (3 * self.img_width // 4 - self.roi_size // 2, self.img_height // 2 - self.roi_size // 2)
    
    def process_image(self, image_data):
        img = image_data.reshape(self.img_height, self.img_width, 3)

        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)

        def get_roi_value(roi_pos):
            x, y = roi_pos
            roi = binary[y:y+self.roi_size, x:x+self.roi_size]
            if roi.size == 0:
                return 0.0
            return np.mean(roi) / 255.0
        left_value = get_roi_value(self.roi_left)
        center_value = get_roi_value(self.roi_center)
        right_value = get_roi_value(self.roi_right)
        
        self.debug_image = binary.copy()
      
        for roi_pos, color in [(self.roi_left, 100), (self.roi_center, 150), (self.roi_right, 200)]:
            x, y = roi_pos
            cv2.rectangle(self.debug_image, 
                         (x, y), 
                         (x + self.roi_size, y + self.roi_size), 
                         color, 1)
        
        return left_value, center_value, right_value


