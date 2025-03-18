"""
File: utils.py
Description: 共通関数の定義ファイル
Author: Kenta Matsumura
Created: 2024-03-17
Last Modified: 2024-03-18
"""

#########################
# ライブラリのインポート
#########################
import glob
import os
import cv2
from constants import *
import pandas as pd

def get_image_paths(folder_path, prefix='', file_extension='png'):
    """
    連番になっている画像ファイルのパスをリストで取得する関数

    Args:
        folder_path (str)   : 画像が格納されているフォルダのパス
        prefix (str)        : 画像ファイルの接頭辞
        file_extension (str): 画像ファイルの拡張子

    Returns:
        image_paths (list of str): ソートされた画像ファイルのパスリスト
    """
    # ファイル名のパターンを設定
    pattern = f"{prefix}*.{file_extension}" if prefix else f"*.{file_extension}"
    
    # ファイル番号でソート
    image_paths = sorted(
        glob.glob(os.path.join(folder_path, pattern)),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].lstrip(prefix))
    )

    return image_paths

def annotate_image(image, detections, distances, model_name):
        """
        2Dの物体検出の認識結果を重畳した画像の作成

        Args:
            image (np.ndarray): RGB画像
            detections (list) : 物体検出結果
                - 'orig_img' (np.ndarray): 推論に使った画像
                - 'orig_shape' (tuple)   : 画像のサイズ(height, width)
                - 'boxes' (Boxes)        : バウンディングボックス情報
                    - 'xyxy' (list)      : バウンディングボックスの座標[x1, y1, x2, y2]
                    - 'conf' (tensor)    : 信頼度スコア
                    - 'cls' (tensor)     : クラスID
                - 'names' (dict)         : key:クラスID、value:クラス名の辞書
            distance (list)  : 物体までの距離の推定結果
            model_name (str) : モデルの名前
        
        Returns:
            annotated_image: 検出結果を重畳した画像
        """

        annotated_image = image.copy()

        if len(distances) < 0: # 車 or 人が画像内にいない場合、何もしない
            pass
        else:
            if model_name == 'YOLO':
                for i, detection in enumerate(detections):
                    for box in detection.boxes:
                        class_id   = int(box.cls.item())           # クラスID
                        class_name = detection.names[class_id]     # クラス名
                        
                        if class_name not in TARGET_CLASSES:
                            continue
                        
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int) # バウンディングボックスの座標
                        confidence = box.conf.item()               # 信頼度スコア
                        distance   = distances[i]                  # 距離(世界座標)

                        # バウンディングボックスの描画
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), COLOR_MAP[class_name], 2)

                        # 認識情報の描画
                        label = f"{class_name}({int(confidence*100)}%) : {distance:.2f}[m]"
                        cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_MAP[class_name], 2)
            else:
                ValueError(f"Invalid model_name : '{model_name}'.")
        
        return annotated_image

def save_annotated_image(annotated_image, raw_image_path, output_path):
    """
    認識結果を重畳した画像を保存

    Args:
        annotate_image (np.ndarray): 重畳画像
        raw_image_path (str)       : 元の画像のパス
        output_path (str)          : 重畳画像の保存先パス   
    """

    # 元の画像のパスの保存先フォルダ名まで取得
    raw_image_path_split = raw_image_path.split('/')
    print(f"raw_image_path:{raw_image_path}, split:{raw_image_path_split}")
    
    if len(raw_image_path_split) >= 2:
        file_name = '/'.join(raw_image_path_split[-2:]) # folder//file.pngを抽出
    else:
        raise ValueError("format of path is wrong")
    
    # 保存先のパス
    print(f"output_path:{output_path}, file_name:{file_name}")
    output_file_name = output_path + file_name

    # 画像を保存
    cv2.imwrite(output_file_name, annotated_image)

    print(f"saved to '{output_file_name}'")

def append_detections_to_df(df, detections, distances, frame):
    """
    認識結果をdfに追加

    Args:
        df (pd.DataFrame)        : 認識結果のデータフレーム
        distances (list of float): 物体までの距離のリスト   
    
    Returns:
        updated_df (pd.DataFrame): 認識結果を追加したデータフレーム
    """

    append_data = [] # 追加するデータ
    if len(distances) <= 0: # 認識結果が0   
        append_data.append([frame, 0, "None", 0, 0, 0, 0, 0])
    for i, detection in enumerate(detections):
        for box in detection.boxes:
            class_id   = int(box.cls.item())           # クラスID
            class_name = detection.names[class_id]     # クラス名
            
            if class_name not in TARGET_CLASSES:
                continue
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int) # バウンディングボックスの座標
            confidence = box.conf.item()               # 信頼度スコア
            distance   = distances[i]                  # 距離(世界座標)

            # 追加データリストに格納
            append_data.append([frame, distance, class_name, confidence, x1, y1, x2, y2])

    # DataFrame に一括追加
    updated_df = pd.concat([df, pd.DataFrame(append_data, columns=df.columns)], ignore_index=True)

    return updated_df

def save_detections_to_csv(df, raw_image_path, output_path):
    """
    認識結果を重畳した画像を保存

    Args:
        df (pd.Dataframe)   : 認識結果のデータフレーム
        raw_image_path (str): 元の画像のパス
        output_path (str)   : csvの保存先パス   
    """

    # 元の画像のパスの保存先フォルダ名を取得
    raw_image_path_split = raw_image_path.split('/')
    if len(raw_image_path_split) >= 2:
        folder_name = raw_image_path_split[-2]
    else:
        raise ValueError("format of path is wrong")
    
    # 保存先のパス
    output_file_name = output_path + folder_name + "detections.csv"

    # csvを保存
    df.to_csv(output_file_name)
