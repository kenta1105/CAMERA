"""
File: DepthEstimator.py
Description: 深度推定を実施するクラス
Author: Kenta Matsumura
Created: 2024-03-13
Last Modified: 2024-03-17
"""

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

#########################
# クラスの定義
#########################
class DepthEstimator:
    """ 深度推定を実施するクラス """

    def __init__(self, model_name, model_ver):
        """
        学習済みモデルの読み込みと画像の変換方法を定義

        Args:
            model_name (str): モデルの名前
            model_ver (str) : 学習済みモデルの種類
        """

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # モデルの読み込み
        if model_name == 'DepthAnythingV2':
            # バージョンの確認
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }
            if model_ver not in model_configs:
                raise ValueError(f"Invalid model version '{model_ver}'. Expected one of '{list(model_configs.keys())}'.")
            
            # 学習済みモデルのアーキテクチャを定義
            self.model = DepthAnythingV2(**model_configs[model_ver])
            
            # 学習済みの重みの読み込み
            try:
                self.model.load_state_dict(torch.load(f'depth_anything_v2_{model_ver}.pth', map_location='cpu'))
            except FileNotFoundError:
                raise FileNotFoundError(f"Moedl file 'depth_anything_v2_{model_ver}.pth' not found.")
            except Exception as e:
                raise RuntimeError(f"Failed to load model weights: {str(e)}")
            
            # 画像の変換方法を定義
            self.transform = Compose([
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
        else:
            raise ValueError(f"Invalid model name : '{model_name}'.")
        
        self.model.to(self.device).eval() # 推論モードに設定

    def predict(self, raw_image):
        """
        深度マップの作成

        Args:
            raw_image (np.ndarray): RGB画像
        
        Returns:
            depth_map (np.ndarray): 深度マップ
        """

        raw_image = raw_image / 255 # 0~1に正規化
        h, w = raw_image.shape[:2]
        image = self.transform({'image': raw_image})['image'] # tranformがDepthAnythingのカスタム定義なので、辞書で渡す
        image = torch.from_numpy(image).unsqueeze(0).to(self.device) # バッチ次元を追加して、tensor化

        with torch.no_grad():
            depth_map = self.model(image) # 推論の実施

        # 補間を使用して、深度マップを元の画像サイズに拡大
        depth_map = F.interpolate(depth_map[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth_map = depth_map.cpu().numpy()

        return depth_map
    
    def save_depth_map(self, depth_map, output_image_name, output_path):
        """
        深度マップの保存

        Args:
            depth_map (np.ndarray) : 深度マップ
            output_image_name (str): 保存する画像の名前
            output_path (str)      : 保存先のパス
        """ 

        # 0~255に正規化
        normalized_depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        normalized_depth_map = (normalized_depth_map * 255).astype(np.uint8)

        # カラーマップで保存
        colored_depth_map = cv2.applyColorMap(normalized_depth_map, cv2.COLORMAP_JET)
        cv2.imwrite(output_path + output_image_name, colored_depth_map)