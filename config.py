import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "line_follower.xml")

# Параметры изображения
IMG_WIDTH = 64
IMG_HEIGHT = 64

# Параметры управления
BASE_SPEED = 0.3
TURN = 0.2
LINE_THRESHOLD = 0.4
MAX_NO_LINE = 150