"""
File: YOLO.py
Description: YOLOの実装クラス
Author: Kenta Matsumura
Created: 2024-03-17
Last Modified: 2024-03-17
"""

#########################
# ライブラリのインポート
#########################
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from constants import TARGET_CLASSES, COLOR_MAP
import torch

#########################
# クラスの定義
#########################
class YOLO:
    """ YOLOの実装クラス """

    def __init__(self, model_name):
        """
        学習済みモデルの読みこみ

        Args:
            model_name (str): 学習済みモデルの名前
        """

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 学習済みの重みの読み込み
        model_path = f'../model/YOLO/{model_name}'
        try:
            self.model = YOLO(model_name)
        except FileNotFoundError:
            raise FileNotFoundError(f"Moedl file '{model_name}' not found.")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {str(e)}")
        
        self.model.to(self.device).eval() # 推論モードに設定

    def predict(self, image):
        """
        2Dの物体検出の実施

        Args:
            image (np.ndarray): RGB画像
        
        Returns:
            detections (list): 物体検出結果
                - 'orig_img' (np.ndarray): 推論に使った画像
                - 'orig_shape' (tuple)   : 画像のサイズ(height, width)
                - 'boxes' (Boxes)        : バウンディングボックス情報
                    - 'xyxy' (list)      : バウンディングボックスの座標[x1, y1, x2, y2]
                    - 'conf' (float)     : 信頼度スコア
                    - 'cls' (float)      : クラスID
                - 'names' (dict)         : key:クラスID、value:クラス名の辞書
        """

        detections = model.predict(raw_image)

        return detections
