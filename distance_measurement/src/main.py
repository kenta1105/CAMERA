#########################
# ライブラリのインポート
#########################
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
import matplotlib.pyplot as plt
from DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2
from DepthAnythingV2.depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
import time



####################
# deviceの設定(CUDA)
####################
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

##########################
# 学習済みモデルの読み込み
##########################
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
encoder = 'vits' # or 'vits', 'vitb', 'vitg'
depth_anything_v2 = DepthAnythingV2(**model_configs[encoder])
depth_anything_v2.load_state_dict(torch.load(f'depth_anything_v2_{encoder}.pth', map_location='cpu'))
depth_anything_v2 = depth_anything_v2.to(DEVICE).eval()

###########
# 深度推定
###########
# Define image transformation patterns
transform = Compose([
    Resize(
        width=518,  # resize it
        height=518,
        resize_target=False,
        keep_aspect_ratio=True, # Maintains the original aspect ratio of the image
        ensure_multiple_of=14,  # Ensures the dimensions are multiples of 14
        resize_method='lower_bound', # Method used for resizing if aspect ratio changes
        image_interpolation_method=cv2.INTER_CUBIC, # Interpolation method used during resizing
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalizes the image with specific mean and standard deviation values
    PrepareForNet(), # Prepares the image for input to a neural network
])
filename = "../data//000000.png"
raw_image = cv2.imread(filename)
raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0 # 0~1に正規化

h, w = raw_image.shape[:2]
image = transform({'image': raw_image})['image']
image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    depth = depth_anything_v2(image) # Obtain depth map from the model

# Resize the depth map to the original image dimensions
depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0] # 深度マップをもとの画像の大きさに補間しながらリサイズ
print(f"min_depth:{depth.min().cpu().numpy()}, max_depth:{depth.max().cpu().numpy()}")

#######################
# 深度マップ⇛3次元点群
#######################


# depth_min = depth.min().cpu().numpy()
# depth_max = depth.max().cpu().numpy()
# depth = depth.cpu()
# depth = (depth - depth_min) / (depth_max - depth_min)
# print(depth.shape)

# # Visualize the depth result
# plt.figure(figsize=(10,6))
# plt.subplot(1,2,1)
# plt.imshow(raw_image)

# plt.subplot(1,2,2)
# plt.imshow(depth, cmap='inferno')
# plt.show()

# with torch.no_grad():
#     start = time.time()
#     depth_map = depth_anything_v2.infer_image(raw_image)
#     end = time.time()
#     print(f"実行時間: {end - start:.4f} 秒")

# # 深度画像を正規化して保存
#     depth_min = depth_2.min()
#     depth_max = depth_2.max()
#     depth_norm = (depth_2 - depth_min) / (depth_max - depth_min + 1e-8)  # 正規化
#     depth_image = (depth_norm * 65535).astype(np.uint16)  # 16ビットに変換

#     save_path = "../data/depth_output.png"
#     cv2.imwrite(save_path, depth_image)
#     print(f"深度画像を保存しました: {save_path}")