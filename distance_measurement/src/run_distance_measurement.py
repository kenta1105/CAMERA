"""
File: run_distance_measurement.py
Description:C2カメラの測距評価用ファイル
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


from constants import *
from ultralytics import YOLO
from DistanceEstimator import DistanceEstimator
import argparse
from utils import *
from constants import *
import pandas as pd

#########################
# main関数
#########################
def main(mode):
    # モデルの定義
    object_detector_2d = YOLO('yolo11x.pt') # 2Dの物体検出モデル
    distance_estimator = DistanceEstimator(CAMERA_INTRINSIC_PARAMS_KITTI, CAMERA_EXTRINSIC_PARAMS_KITTI, 'KITTI') # 物体の距離を推定するクラス

    # オンライン or オフラインで画像の読み先が変わる
    if mode == 'online': # オンラインでの処理
        pass

    elif mode == 'offline': # オフラインでの処理
        image_paths = get_image_paths(IMAGE_PATH)
        frame = 0 # フレーム番号
        df = pd.DataFrame(columns=['frame', 'distance', 'class', 'confidence', 'x1', 'y1', 'x2', 'y2'])
        for image_path in image_paths:
            raw_image = cv2.imread(image_path)
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) 

            # 歪み補正するならここに処理書く

            # 2Dの物体検出
            detections = object_detector_2d.predict(raw_image)

            # 距離の推定
            distances = distance_estimator.estimate_distance(detections, 'YOLO')

            # 画像に認識結果を重畳表示
            annotated_image = annotate_image(raw_image, detections, distances, 'YOLO')

            # 重畳画像を保存
            save_annotated_image(annotated_image, image_path, OUTPUT_PATH)

            # 認識結果と距離をdfに追加
            df = append_detections_to_df(df, detections, distances, frame)
        
        # 認識結果をcsvに保存
        save_detections_to_csv(df, image_path, OUTPUT_PATH)

    else:
        raise ValueError("Invalid mode : '{mode}'. Please specify 'online' of 'offline'.")
    
if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description="Performs distance estimation in either online or offline mode.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["online", "offline"],
        required=True,
        help="Specify the execution mode. Choose either 'online' or 'offline'."
    )
    args = parser.parse_args()

    # main関数の実行
    main(args.mode)