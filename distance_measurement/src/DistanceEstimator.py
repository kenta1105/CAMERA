"""
File: DistanceEstimator.py
Description: 物体の距離を推定するクラス
Author: Kenta Matsumura
Created: 2024-03-13
Last Modified: 2024-03-17
"""

#########################
# ライブラリのインポート
#########################
import numpy as np
from constants import *

#########################
# クラスの定義
#########################
class DistanceEstimator:
    """ 物体の見かけの大きさと既知の大きさを元に、物体の距離を推定するクラス """

    def __init__(self, intrinsic_params, extrinsic_params, sensor_type):
        """
        クラス変数の初期化

        Args:
            intrinsic_params (dict): カメラの内部パラメータ
                - 'camera_matrix' (np.ndarray)          : カメラ行列
                - 'dist_coeffs' (np.ndarray)            : 歪み係数
                - 'image_size' (tuple of int)           : 歪み補正前の画像サイズ
                - 'rectified_image_size' (tuple of int) : 歪み補正後の画像サイズ

            extrinsic_params (dict): カメラの外部パラメータ
                - 'rotation_matrix' (np.ndarray)    : 歪み補正前の回転行列
                - 'translation_vector' (np.ndarray) : 歪み補正前の並進ベクトル
                - 'rectified_rotation_matrix'       : 歪み補正後の回転行列
                - 'projection_matrix'               : 歪み補正後の投影行列             
        """

        if sensor_type == "KITTI":
            # 内部パラメータ
            self.fx = extrinsic_params['projection_matrix'][0, 0]
            self.fy = extrinsic_params['projection_matrix'][1, 1]
            self.cx = extrinsic_params['projection_matrix'][0, 2]
            self.cy = extrinsic_params['projection_matrix'][1, 2]

            # 外部パラメータ
            self.R = extrinsic_params['rectified_rotation_matrix'] # (3, 3)
            Tx = extrinsic_params['projection_matrix'][0, 3] / self.fx
            Ty = extrinsic_params['projection_matrix'][1, 3] / self.fy
            Tz = extrinsic_params['projection_matrix'][2, 3] 
            self.T = np.array([Tx, Ty, Tz]).reshape(3, 1) # (3, 1)

        elif sensor_type == "C2":
            pass
        else:
            ValueError(f"Invalid sensor_type : '{sensor_type}'.")

    def estimate_distance(self, detections, model_name):
        """
        見かけ上の物体の高さと既知の物体の高さを基に、物体までの距離を推定

        Args:
            detections (list of object): 物体認識結果
            model_name (str)           : モデルの名前(物体認識結果の形式がモデルごとに異なる)

        Returns:
            distances (list of float) : 物体までの距離
        """
        
        distances = [] # 物体までの距離

        if model_name == 'YOLO':
            for detection in detections:
                for box in detection.boxes:
                    # 検出した物体が、距離の推定対象の物体(車 or 歩行者)かどうかの判定
                    class_id   = int(box.cls.item())           # クラスID
                    class_name = detection.names[class_id]     # クラス名
                    
                    if class_name not in TARGET_CLASSES:
                        continue

                    # 既知の高さと見かけの高さの取得
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy() # バウンディングボックスの座標
                    known_height = OBJECT_SIZE[class_name] # 既知の物体の高さ
                    observed_height = y2 - y1 # 画像上の高さ

                    # 0割防止
                    if observed_height <= 0:
                        raise ValueError("Observed height must be greater than zero.")

                    # 距離の推定
                    distance = (self.fy * known_height) / observed_height
                    distances.append(distance)
        else:
            ValueError(f"Invalid model_name : '{model_name}'.")

        return distances


