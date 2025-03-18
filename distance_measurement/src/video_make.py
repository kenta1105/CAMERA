import cv2
import glob
from utils import *
from constants import *

IMAGE_PATH = "../output/test"
# 画像ファイルを取得
image_paths = get_image_paths(IMAGE_PATH)

# 動画の設定
frame = cv2.imread(image_paths[0])
h, w, _ = frame.shape
fps = 1  # フレームレート

# 動画を保存するための設定
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4フォーマット
out = cv2.VideoWriter('output_dist_1FPS.mp4', fourcc, fps, (w, h))

# 画像を順番に動画に追加
for image_path in image_paths:
    frame = cv2.imread(image_path)
    out.write(frame)

out.release()
cv2.destroyAllWindows()
