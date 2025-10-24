#!/usr/bin/env python
from PIL import Image, ImageOps
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage import measure
import matplotlib.pyplot as plt
from scipy import interpolate
import os
import glob
import csv
import pandas as pd
import math
import statistics
import PIL
from ultralytics import YOLO
import shutil
import time

# サンプルデータ
class Sample():
    def __init__(self, condition, sample_path, span, magnification, setting):
        self.condition = condition
        self.sample_path = sample_path # sample01/
        self.pattern_list = ['pattern01', 'pattern02']
        self.magnification = magnification
        self.size = [] # サンプルの画像サイズ
        self.vessel_data = []
        self.branch_data = []
        self.span = span
        self.setting = setting
        self.set_dir()
        self.trim(span) 
        # こっからループを2つ回したい
        for pattern in self.pattern_list:
            print()
            self.preprocess(pattern)
            self.segmentation_all_img(pattern)
            ImageProcessor(self, pattern) # ここでセグメントされた画像のディレクトリを受け取る

        # csv_dataからcsvを作成
        with open(f"{self.sample_path}/vessel_result.csv", mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(['sample', 'pattern', 'name', 'vessel_id', 'branches', 'length', 'thickness', 'qualified', 'thickness_list', 'modified_x', 'modified_y', 'modified_forward_x', 'modified_forward_y', 'modified_reverse_x', 'modified_reverse_y'])
            writer.writerows(self.vessel_data)

        # csv_dataからcsvを作成
        with open(f"{self.sample_path}/branch_result.csv", mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(['sample', 'pattern', 'name', 'branch_id', 'branch_position', 'vessels', 'modified_x', 'modified_y'])
            writer.writerows(self.branch_data)        
    
    
    def set_dir(self):
        os.makedirs(f"{self.sample_path}/01 raw", exist_ok=True)
        os.makedirs(f"{self.sample_path}/pattern01", exist_ok=True)
        os.makedirs(f"{self.sample_path}/pattern02", exist_ok=True)
        for pattern in self.pattern_list:        
            os.makedirs(f"{self.sample_path}/{pattern}/02 trimmed", exist_ok=True)
            os.makedirs(f"{self.sample_path}/{pattern}/03 preprocessed", exist_ok=True)
            os.makedirs(f"{self.sample_path}/{pattern}/04 segmented", exist_ok=True)
            os.makedirs(f"{self.sample_path}/{pattern}/04 segmented/segmented", exist_ok=True)
            os.makedirs(f"{self.sample_path}/{pattern}/04 segmented/binary", exist_ok=True)
            os.makedirs(f"{self.sample_path}/{pattern}/05 result", exist_ok=True)


    def trim(self, span):

        files = os.listdir(f"{self.sample_path}/01 raw/")
        for file in files:
            print(file)
            img_path = os.path.join(self.sample_path, '01 raw', file)
            break
        print(self.sample_path)
        print(img_path)
        img = Image.open(img_path)

        # 画像のサイズ取得
        w, h = img.size
        print(f"元のサイズ: {w}x{h}")

        # 余白を計算
        w_room = (span - (w % span)) % span
        h_room = (span - (h % span)) % span

        # 余白を右下に追加（黒背景）
        new_w = w + w_room
        new_h = h + h_room
        new_img = ImageOps.expand(img, (0, 0, w_room, h_room), fill=(0, 0, 0))

        # 保存
        os.makedirs(f"{self.sample_path}/01 raw/scaled", exist_ok=True)
        output_path = os.path.join(self.sample_path, "01 raw/scaled/", file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        new_img.save(output_path)

        img_num = 0

        w_roop = w//span+1
        h_roop = h//span+1

        for h_index in range(h_roop):
            for w_index in range(w_roop):
                crop_w = span*w_index
                crop_h = span*h_index
                im_crop = new_img.crop((crop_w, crop_h, crop_w+span, crop_h+span))
                im_crop.save(f"{self.sample_path}/{self.pattern_list[0]}/02 trimmed/cropped_{img_num}_w{w_index}_h{h_index}.png")
                img_num += 1

        print("=================================================================")

        img_num = 0
        for h_index in range(h_roop-1):
            for w_index in range(w_roop-1):
                crop_w = int(span*w_index + span/2)
                crop_h = int(span*h_index + span/2)
                im_crop = new_img.crop((crop_w, crop_h, crop_w+span, crop_h+span))
                im_crop.save(f"{self.sample_path}/{self.pattern_list[1]}/02 trimmed/cropped_{img_num}_w{w_index}_h{h_index}.png")
                img_num += 1
    
    def preprocess(self, pattern):
        files = glob.glob(f"{self.sample_path}/{pattern}/02 trimmed/*")
        for file in files:
            grayscale_img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            averaged_img = cv2.blur(grayscale_img, (5, 5))
            standardized_img = cv2.normalize(averaged_img, None, 0, 255, cv2.NORM_MINMAX)  # 標準化
            cv2.imwrite(f"{self.sample_path}/{pattern}/03 preprocessed/{file.split('/')[-1]}", standardized_img)  # 処理した画像を保存

    def segmentation(self, image_path, pattern):
        # セグメンテーションを行う画像のパス
        image_id = image_path.split("/")[-1].split(".")[0].replace("cropped_", "")  # 画像IDを取得
        # 画像に対してセグメンテーションを実行
        results = self.setting.model(image_path)

        # 結果を表示
        plt.figure(figsize=(8, 8))
        for i, r in enumerate(results):  # インデックスを追加
            im_array = r.plot()
            plt.imshow(cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            # 結果を保存
            plt.savefig(f'{self.sample_path}/{pattern}/04 segmented/segmented/{image_id}.png', bbox_inches='tight', pad_inches=0)  # 結果を保存

            # 結果を2値マスクとして表示
        plt.figure(figsize=(8, 8))
        for i, r in enumerate(results):  # インデックスを追加
            # セグメンテーションマスクを取得
            if r.masks is not None:  # dataがない場合はスキップ
                masks = r.masks.data.cpu().numpy()
            else:
                continue 
            
            # すべてのマスクを組み合わせて1つの2値マスクを作成
            combined_mask = np.any(masks, axis=0).astype(np.uint8)
            
            # マスクを反転（検出部分を黒、それ以外を白に）
            binary_mask = 1 - combined_mask
            
            # マスクを表示
            plt.imshow(binary_mask, cmap='gray')
            plt.axis('off')
            # マスクを保存
            plt.savefig(f'{self.sample_path}/{pattern}/04 segmented/binary/{image_id}.png', bbox_inches='tight', pad_inches=0)  # マスクを保存

    def segmentation_all_img(self, pattern):
        files = glob.glob(f"{self.sample_path}/{pattern}/03 preprocessed/*")    
        for file in files:
            self.segmentation(file, pattern)


# 現状はセグメントされた画像のディレクトリを受け取る
class ImageProcessor():
    def __init__(self, sample, pattern):
        self.sample = sample
        self.setting = sample.setting
        self.sample_path = self.sample.sample_path
        self.pattern = pattern
        self.img_dir = f"{self.sample_path}/{pattern}/04 segmented/binary"
        self.process_all_images()
    def process_all_images(self):
        image_files = glob.glob(os.path.join(self.img_dir, "*.png"))
        for img_path in image_files:
            SegmentedImage(img_path, self.sample, self.pattern)

# セグメントされた画像を受け取る
class SegmentedImage():
    def __init__(self, img_path, sample, pattern):
        self.path = img_path
        self.sample = sample
        self.setting = sample.setting
        self.sample_path = self.sample.sample_path
        self.pattern = pattern
        self.name = os.path.basename(img_path).split('.')[0]
        self.img_data = cv2.imread(img_path)
        self.gray_data = cv2.cvtColor(self.img_data, cv2.COLOR_BGR2GRAY)
        self.height, self.width, _ = self.img_data.shape
        self._, self.binary = cv2.threshold(self.gray_data, 128, 255, cv2.THRESH_BINARY)
        self.binary_norm = self.binary.astype(np.float32) / 255.0 # distanceを測っている画像がこれ
        self.scale = self.binary_norm.shape[0] # binary_normのwidth, heightを取得
        self.span = self.sample.span # input画像のspanを取得
        self.magnification = self.sample.magnification # 倍率
        self.adjust_scale = self.span * self.magnification / self.scale # distanceにかける値
        self.skeleton = skeletonize(self.binary_norm)
        self.branch_points = self.get_branch_points()
        self.deleted_branch = self.delete_branch_points()
        self.vessels_list = self.get_vessels()
        self.branch_point_obj_list = self.get_branch_obj_list() # 結合している血管の情報が入ったBranchPointオブジェクトのリスト     
        self.update_vessels()
        self.visualize = Visualize(self)
        # セグメントされた画像サイズと、元のspanの比を考える必要がある
        CSVWriter(self, self.sample)
        self.result_img = self.visualize.result_img

    def get_branch_points(self):
        branch_points = []
        branch_dict = {}
        connected_points = []
        # この時、接合している点の情報も追加
        for i in range(1, self.skeleton.shape[0] - 1):
            for j in range(1, self.skeleton.shape[1] - 1):
                if self.skeleton[i, j]:
                    neighbors = np.sum(self.skeleton[i-1:i+2, j-1:j+2]) - 1
                    if 3 <= neighbors <= 5:
                        frame_list = [self.skeleton[i-1, j-1], self.skeleton[i-1, j], self.skeleton[i-1, j+1], self.skeleton[i, j+1], self.skeleton[i+1, j+1], self.skeleton[i+1, j], self.skeleton[i+1, j-1], self.skeleton[i, j-1]]
                        count = 0
                        status = False
                        for frame in frame_list:
                            if status == False and frame == True:
                                count += 1
                            status = frame
                        if frame_list[0] == True and frame_list[-1] == True:
                            count -= 1
                        if count >= 3:
                            branch_points.append((i, j))
        return branch_points
    
    def get_branch_obj_list(self):
        branch_point_obj_list = []
        print(self.branch_points)
        for id, branch_point in enumerate(self.branch_points):
            connecting_vessels = []
            x, y = branch_point
            frame_list = [
                (x-2, y-2),
                (x-2, y-1),
                (x-2, y),
                (x-2, y+1),
                (x-2, y+2),
                (x-1, y-2),
                (x-1, y+2),
                (x, y-2),
                (x, y+2),
                (x+1, y-2),
                (x+1, y+2),
                (x+2, y-2),
                (x+2, y-1),
                (x+2, y),
                (x+2, y+1),
                (x+2, y+2)
            ]

            # スケルトン画像のサイズを取得
            height, width = self.skeleton.shape

            for frame in frame_list:
                # 境界チェックを追加
                if 0 <= frame[0] < height and 0 <= frame[1] < width:
                    # 範囲内のときのみ処理を実行
                    if self.skeleton[frame[0], frame[1]]:
                        for vessel in self.vessels_list:
                            if frame in vessel.points:
                                connecting_vessels.append(vessel)
                else:
                    # 範囲外のフレームを無視
                    print(f"Skipped frame {frame} (out of bounds)")

            # ブランチポイントオブジェクトを作成
            branch_point_obj = BranchPoint(f'branch{id}', branch_point, connecting_vessels)
            branch_point_obj_list.append(branch_point_obj)

        return branch_point_obj_list

    def update_vessels(self):
        for branch_point in self.branch_point_obj_list:
            for vessel in branch_point.connecting_vessels:
                vessel.branches.append(branch_point)
                vessel.branch_ids.append(branch_point.id)
    

    def delete_branch_points(self):
        # 分岐点の消去, 周囲の点も削除
        skel_copy = self.skeleton.copy()
        for point in self.branch_points:
            skel_copy[point[0]-1:point[0]+2, point[1]-1:point[1]+2] = 0
        return skel_copy

    def get_vessels(self):
        # 結合している点順に取得
        def find_connected_points(coordinates):
            def is_adjacent(p1, p2):
                return abs(p1[0] - p2[0]) <= 1 and abs(p1[1] - p2[1]) <= 1 and p1 != p2

            def is_blocking(p, coord):
                return p[0] == coord[0] or p[1] == coord[1]

            connected_points = {}

            for coord in coordinates:
                connected = []
                blocking_points = [p for p in coordinates if is_blocking(p, coord) and p != coord]
                
                for point in coordinates:
                    if point == coord:
                        continue
                    if is_adjacent(coord, point):
                        if point in blocking_points:
                            connected.append(point)
                        else:
                            is_blocked = False
                            for blocking_point in blocking_points:
                                if is_blocking(blocking_point, point):
                                    is_blocked = True
                                    break
                            if not is_blocked:
                                connected.append(point)

                connected_points[tuple(coord)] = connected
            return connected_points

        # 再帰的に結合している点を探す  
        def find_connection(start, point_dict, vessel_points, end, depth=0, max_depth=500):
            """ 再帰の深さ制限を設定し、無限ループを防ぐ """
            if depth >= max_depth:
                print(f"Max recursion depth {max_depth} reached.")
                return
            
            print(f"Depth: {depth}, Start: {start}")

            # point_dictのキーをコピーしてループ
            keys_to_check = list(point_dict.keys())

            for i in keys_to_check:
                if list(start) in point_dict[i]:
                    del point_dict[i]  # pop() ではなく del を使う
                    start = i
                    vessel_points.append(i)

                    # 終了条件
                    if start == end:
                        return
                    
                    # 次の再帰呼び出し
                    find_connection(start, point_dict, vessel_points, end, depth + 1, max_depth)
                    break  # 1つのルートを辿るなら break で終了


        # ラベリング：分割
        labeled, num_features = measure.label(self.deleted_branch, return_num=True, connectivity=2)
        vessels = []
        for i in range(num_features):
            vessel_name = 'vessel' + str(i + 1)
            coordinates = np.argwhere(labeled == i + 1) # labelの部分の座標を取得
            coordinates_list = coordinates.tolist()
            result = find_connected_points(coordinates_list) # 結合を取得
            point_dict = result.copy()
            edge_list = []
            for i in result:
                if len(result[i]) == 1:
                    edge_list.append(i)

            if len(edge_list) >= 2:
                start = edge_list[0]
                end = edge_list[1]
                vessel_points = [start]
                point_dict.pop(start)
                find_connection(start, point_dict, vessel_points, end)
                if len(vessel_points) > 2: # さすがに
                    vessel = Vessel(self, vessel_name, vessel_points, self.setting) # vesselクラスのインスタンスを生成
                    vessels.append(vessel)
        return vessels # Vesselのリストを返す
    
# 血管
class Vessel():
    def __init__(self, img, vessel_name, vessel_points, setting):
        self.setting = setting
        self.belong_img = img
        self.name = vessel_name
        self.id = int(vessel_name.split('vessel')[1])
        self.points = vessel_points # Vesselを構成する点の座標リスト
        self.length = len(self.points)
        self.point_object_list = self.get_point_list() # 中央部分のpointオブジェクトのリスト
        self.center_point = self.point_object_list[len(self.point_object_list)//2] # 中央のpointオブジェクト
        self.thickness = self.center_point.distance
        self.thickness_list = self.get_thickness_list()
        self.qualified = self.is_qualified() # code
        self.branches = [] # branchオブジェクトのリスト
        self.branch_ids = []

    def get_point_list(self):
        first_quarer = self.length//4
        third_quarer = self.length*3//4
        point_list = []
        for index, point in enumerate(self.points):
            # 中央部分のみ処理
            if first_quarer <= index < third_quarer:
                point = Point(self, index)
                point_list.append(point)
        return point_list
    
    def get_thickness_list(self):
        distance_list = [point.distance for point in self.point_object_list if point.distance is not None]
        return distance_list
    
    def is_qualified(self):
        code = -1
        # distanceがない場合: code 1
        if not self.center_point.distance:
            code = 1
            return code

        # 中央点の左右差えぐいパターン: code 2
        def get_distance(point1, point2):
            distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
            return distance
        
        forward_distance = get_distance(self.center_point.position, self.center_point.distance_points[0])
        reverse_distance = get_distance(self.center_point.position, self.center_point.distance_points[1])
        distance_ratio = forward_distance / reverse_distance
        threshold = self.setting.distance_ratio
        min_length = self.setting.min_length_percentage * self.belong_img.width
        max_length = self.setting.max_length_percentage * self.belong_img.width
        if distance_ratio > threshold or distance_ratio < 1/threshold:
            code = 2
            return code
        
        # 長さが設定範囲内かどうか: code 3
        if self.length < min_length or self.length > max_length:
            code = 3
            return code

        # 分散がいかちいパターン: code 4
        distance_list = [point.distance for point in self.point_object_list if point.distance is not None]        
        variance = statistics.variance(distance_list) if len(distance_list) > 1 else 0
        if variance > self.setting.variance_threshold:
            code = 4
            return code
        
        # 長さと太さの比がおかしいやつ： code 5
        thickness = self.center_point.distance
        thickness_length_ratio = thickness / self.length
        if thickness_length_ratio > 1:
            code = 5
            return code

        return 0

# 点を取得
class Point():
    def __init__(self, vessel, point_index):
        self.setting = vessel.setting
        self.position = vessel.points[point_index]
        self.belong_vessel = vessel
        self.belong_img = vessel.belong_img
        self.point_index = point_index
        self.direction = self.get_direction()
        self.distance_points = self.get_points() # [forward_point, reverse_point]
        self.distance = self.get_distance()

    def get_direction(self):
        range = self.setting.range
        # 法線and単位ベクトルに変換する関数を用意
        def simplify_vector(x, y):
            if x == 0 and y == 0:
                return [0, 0]
            length = math.sqrt(x**2 + y**2)
            x = x / length
            y = y / length
            return [-y, x]

        # pointに対する法線を計算する
        # rangeよりも大きいことを確認
        if range < self.point_index < self.belong_vessel.length - range:
            x1, y1 = self.belong_vessel.points[self.point_index - range]
            x2, y2 = self.belong_vessel.points[self.point_index + range]
            x = x2 - x1
            y = y2 - y1
            return simplify_vector(x, y)
        else:
            return None
        
    def get_points(self):
        if not self.direction:
            return None
        binary_norm = self.belong_img.binary_norm.copy() # distanceを測る対象の画像
        start_point = self.position
        dx, dy = self.direction # 単位ベクトルのはず
        start_x, start_y = start_point

        current_x, current_y = start_x, start_y # 初期設定
        height, width = binary_norm.shape
        found = False
        # まず順行性に探索
        forward_point = None,
        reverse_point = None
        while not found:
            # 座標を更新（浮動小数点のまま計算）
            current_x += dx
            current_y += dy
            ix, iy = round(current_x), round(current_y)  # 小数点以下を四捨五入

            # はみ出ていたら探索終了
            if ix < 0 or ix >= height or iy < 0 or iy >= width:
                return None

            # 目的の点を見つけたら即座に確定
            if binary_norm[ix, iy] == 0:
                forward_point = [ix, iy]
                break  # 無駄な探索を防ぐ

            # 近傍探索（ノイズを拾う可能性を減らす）
            for around_dx in [-1, 0, 1]:
                for around_dy in [-1, 0, 1]:
                    nx, ny = ix + around_dx, iy + around_dy
                    if 0 <= nx < height and 0 <= ny < width:
                        if binary_norm[nx, ny] == 0:
                            forward_point = [nx, ny]
                            found = True
                            break
                if found:
                    break  # 外側のループも抜ける

        # 再び初期化
        found = False
        dx, dy = -dx, -dy
        current_x, current_y = start_x, start_y 
        while not found:
        # 座標を更新（浮動小数点のまま計算）
            current_x += dx
            current_y += dy
            ix, iy = round(current_x), round(current_y)  # 小数点以下を四捨五入

            # はみ出ていたら探索終了
            if ix < 0 or ix >= height or iy < 0 or iy >= width:
                return None

            # 目的の点を見つけたら即座に確定
            if binary_norm[ix, iy] == 0:
                reverse_point = [ix, iy]
                break  # 無駄な探索を防ぐ

            # 近傍探索（ノイズを拾う可能性を減らす）
            for around_dx in [-1, 0, 1]:
                for around_dy in [-1, 0, 1]:
                    nx, ny = ix + around_dx, iy + around_dy
                    if 0 <= nx < height and 0 <= ny < width:
                        if binary_norm[nx, ny] == 0:
                            reverse_point = [nx, ny]
                            found = True
                            break
                if found:
                    break  # 外側のループも抜ける

        # forward, reverseの点ともにある場合
        if forward_point and reverse_point:
            return [forward_point, reverse_point]
        else:
            return None
        
    def get_distance(self):
        if self.distance_points: # 両方の点があるはず 
            forward_point, reverse_point = self.distance_points
            distance = math.sqrt((forward_point[0] - reverse_point[0])**2 + (forward_point[1] - reverse_point[1])**2)
            distance *= self.belong_img.adjust_scale # ここでμmに変換！
            return distance
        return None


class Visualize():
    def __init__(self, segmented_img):
        self.img = segmented_img
        self.sample_path = self.img.sample_path
        self.pattern = self.img.pattern
        self.output_path = f"{self.sample_path}/{self.pattern}/05 result"
        self.vis_img = cv2.cvtColor(self.img.binary_norm.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        self.visualize_with_value()
        self.result_img = self.visualize_with_background()
    
    def visualize_normal(self):
        vis_img = self.vis_img.copy()
        os.makedirs(f'{self.output_path}/result_normal', exist_ok=True)

        # segmented_img のサイズを取得
        height, width = self.img.binary_norm.shape  # (rows, cols) → (height, width)

        # 余白削除用の設定
        plt.clf()
        plt.figure(figsize=(width / 100, height / 100))  # 画像のサイズに応じた figsize
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 画像にフィット

        for vessel in self.img.vessels_list:
            for point in vessel.points:
                vis_img[point[0], point[1]] = [0, 0, 255]  # 赤色

            center_point_position = vessel.center_point.position
            center = (center_point_position[1], center_point_position[0])
            plt.plot(center[0], center[1], 'bo', markersize=2)

            if vessel.center_point.distance:
                forward_point, reverse_point = vessel.center_point.distance_points
                forward = (forward_point[1], forward_point[0])
                reverse = (reverse_point[1], reverse_point[0])
                plt.plot(forward[0], forward[1], 'go', markersize=2)
                plt.plot(reverse[0], reverse[1], 'ro', markersize=2)

                # qualifiedの値によって線の色を変える
                color_map = {
                    0: ('b', 1.0),
                    1: ('g', 0.5),
                    2: ('y', 0.5),
                    3: ('orange', 0.5),
                    4: ('r', 0.5),
                    5: ('purple', 0.5)
                }

                color, alpha = color_map.get(vessel.qualified, ('k', 0.5))  # デフォルトは黒・半透明
                plt.plot([forward[0], reverse[0]], [forward[1], reverse[1]], 
                        color=color, linewidth=2, alpha=alpha)


        # 画像をプロット
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # 軸を非表示
        plt.gca().set_frame_on(False)  # フレームも削除
        plt.gca().set_aspect('equal', adjustable='box')  # アスペクト比維持

        # 余白を完全になくし、dpiを調整して保存
        plt.savefig(
            f'{self.output_path}/result_normal/{self.img.name}.png',
            bbox_inches='tight',
            pad_inches=0,
            dpi=100  # DPIを固定
        )
        plt.close()

    
    def visualize_with_value(self):
        vis_img_with_value = self.vis_img.copy()
        os.makedirs(f'{self.output_path}/result_with_value', exist_ok=True)
        # segmented_img のサイズを取得
        height, width = self.img.binary_norm.shape  # (rows, cols) → (height, width)

        # 余白削除用の設定
        plt.clf()
        plt.figure(figsize=(width / 100, height / 100))  # 画像のサイズに応じた figsize
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 画像にフィット
        for vessel in self.img.vessels_list:
            # 線の全ての点を表示
            for point in vessel.points:
                vis_img_with_value[point[0], point[1]] = [0, 0, 255] # 赤色, matplotlib座標系に合わせる

            # 線の中央点を表示
            center_point_position = vessel.center_point.position
            center = (center_point_position[1], center_point_position[0])
            plt.plot(center[0], center[1], 'bo', markersize=2)
            if vessel.center_point.distance: # distanceが一番厳しい条件
                # 中央点に対して、distance_pointsを表示
                forward_point, reverse_point = vessel.center_point.distance_points
                forward = (forward_point[1], forward_point[0])
                reverse = (reverse_point[1], reverse_point[0])
                plt.plot(forward[0], forward[1], 'go', markersize=2)
                plt.plot(reverse[0], reverse[1], 'ro', markersize=2)

                # qualifiedの値によって色を変えて表示
                if vessel.qualified == 0:
                    plt.plot([forward[0], reverse[0]], [forward[1], reverse[1]], 'b-', linewidth=2) # 青
                elif vessel.qualified == 1:
                    plt.plot([forward[0], reverse[0]], [forward[1], reverse[1]], 'g-', linewidth=2) # 緑
                elif vessel.qualified == 2:
                    plt.plot([forward[0], reverse[0]], [forward[1], reverse[1]], 'y-', linewidth=2) # 黄色
                elif vessel.qualified == 3:
                    plt.plot([forward[0], reverse[0]], [forward[1], reverse[1]], 'orange', linewidth=2) # オレンジ
                elif vessel.qualified == 4:
                    plt.plot([forward[0], reverse[0]], [forward[1], reverse[1]], 'r-', linewidth=2) # 赤

                # 値を表示
                plt.text((forward[0] + reverse[0]) / 2, (forward[1] + reverse[1]) / 2, f'{vessel.thickness:.2f}', color='white', fontsize=7, bbox=dict(facecolor='black', alpha=0.5))

        # 画像をプロット
        plt.imshow(cv2.cvtColor(vis_img_with_value, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # 軸を非表示
        plt.gca().set_frame_on(False)  # フレームも削除
        plt.gca().set_aspect('equal', adjustable='box')  # アスペクト比維持

        # 余白を完全になくし、dpiを調整して保存
        plt.savefig(
            f'{self.output_path}/result_with_value/{self.img.name}.png', 
            bbox_inches='tight',
            pad_inches=0,
            dpi=100  # DPIを固定
        )
        plt.close()


    def visualize_with_background(self):
        result_img = cv2.imread(f'{self.output_path}/result_with_value/{self.img.name}.png')
        background_img = cv2.imread(f'{self.sample_path}/{self.pattern}/03 preprocessed/cropped_{self.img.name}.png')

        # 画像のサイズを一致させる
        if result_img.shape[:2] != background_img.shape[:2]:
            background_img = cv2.resize(background_img, (result_img.shape[1], result_img.shape[0]))

        # 画像を重ねる
        overlay_img = cv2.addWeighted(result_img, 0.5, background_img, 0.5, 0)

        # 重ねた画像を保存
        os.makedirs(f'{self.output_path}/result_with_background', exist_ok=True)
        cv2.imwrite(f'{self.output_path}/result_with_background/{self.img.name}_overlay.png', overlay_img)
        return overlay_img # 重ねた画像を返す

class BranchPoint():
    def __init__(self, name, position, connecting_vessels):
        self.name = name #branch
        self.id = int(name.split('branch')[1])
        self.position = position
        self.connecting_vessels = connecting_vessels

class CSVWriter():
    def __init__(self, segmented_img, sample):
        self.img = segmented_img
        self.sample = sample
        self.sample_path = self.img.sample_path
        self.sample_id = self.img.sample_path.split("/")[-1]
        self.pattern = self.img.pattern
        self.segmented_width = self.img.width
        self.span = self.img.span
        self.output_path = f"{self.sample_path}/{self.pattern}/05 result"
        self.write_csv()
        self.write_branch_csv()
    
    def write_csv(self): #imgごとに回っている
        os.makedirs(f'{self.output_path}/csv', exist_ok=True)
        with open(f'{self.output_path}/csv/{self.img.name}.csv', 'w') as f:
            w_index = int(self.img.name.split("_w")[1][0])
            h_index = int(self.img.name.split("_h")[1][0])
            writer = csv.writer(f)
            writer.writerow(['sample','name','vessel_id', 'length', 'thickness', 'qualified', 'branches', 'thickness_list', 'position', 'modified_position_x', 'modified_position_y'])
            ratio = self.span / self.segmented_width
            span = self.span
            for vessel in self.img.vessels_list:
                branch_set = set(vessel.branch_ids)
                center_x, center_y = vessel.center_point.position
                if vessel.center_point.distance:
                    forward_point, reverse_point = vessel.center_point.distance_points

                    modified_x = w_index*span + center_y*ratio
                    modified_y = h_index*span + center_x*ratio
                    
                    modified_forward_x = w_index*span + forward_point[1]*ratio
                    modified_forward_y = h_index*span + forward_point[0]*ratio

                    modified_reverse_x = w_index*span + reverse_point[1]*ratio
                    modified_reverse_y = h_index*span + reverse_point[0]*ratio

                    if self.pattern == "pattern02":
                        modified_x += span/2
                        modified_y += span/2
                        
                        modified_forward_x += span/2
                        modified_forward_y += span/2

                        modified_reverse_x += span/2
                        modified_reverse_y += span/2
                    
                    modified_x = int(modified_x)
                    modified_y = int(modified_y)
                    
                    modified_forward_x = int(modified_forward_x)
                    modified_forward_y = int(modified_forward_y)

                    modified_reverse_x = int(modified_reverse_x)
                    modified_reverse_y = int(modified_reverse_y)
                    
                    data = [self.sample_id, self.pattern, self.img.name, vessel.id, branch_set, vessel.length, vessel.thickness, vessel.qualified, vessel.thickness_list, modified_x, modified_y, modified_forward_x, modified_forward_y, modified_reverse_x, modified_reverse_y]
                    self.sample.vessel_data.append(data)

                writer.writerow([self.sample_id, self.img.name, vessel.id, vessel.length, vessel.thickness, vessel.qualified, branch_set, vessel.thickness_list, vessel.center_point.position])

    def write_branch_csv(self):
        os.makedirs(f'{self.output_path}/csv_branch', exist_ok=True)
        with open(f'{self.output_path}/csv_branch/{self.img.name}.csv', 'w') as f:
            w_index = int(self.img.name.split("_w")[1][0])
            h_index = int(self.img.name.split("_h")[1][0])
            ratio = self.span / self.segmented_width
            span = self.span
            writer = csv.writer(f)
            writer.writerow(['sample', 'pattern', 'img', ''])
            for branch_point in self.img.branch_point_obj_list:
                position_x, position_y = branch_point.position

                modified_x = w_index*span + position_y*ratio
                modified_y = h_index*span + position_x*ratio

                if self.pattern == "pattern02":
                    modified_x += span/2
                    modified_y += span/2

                vessels_list = []
                for vessel in branch_point.connecting_vessels:
                    vessels_list.append(vessel.id)

                writer.writerow([self.sample_id, self.img.name, branch_point.id, branch_point.position, set(vessels_list)])
                data = [self.sample_id, self.pattern, self.img.name, branch_point.id, branch_point.position, set(vessels_list), modified_x, modified_y]
                self.sample.branch_data.append(data)

def main(setting):
    start_time = time.time()    

    def organize_files(csv_path, project, span):
        """
        CSVファイルを読み込んで、条件ごとにファイルを整理する
        
        Parameters:
        csv_path (str): 入力CSVファイルのパス
        span (int): スパン値
        
        Returns:
        pd.DataFrame: 新しいパスを含むDataFrame
        """
        # CSVファイルを読み込む
        df = pd.read_csv(csv_path)
        
        # 結果を格納する新しい列を作成
        new_paths = []
        
        # 条件ごとにファイルを処理
        # "condition"という文字列の行をスキップ
        for condition in df[df['condition'] != 'condition']['condition'].unique():
            # 条件ごとのデータを取得
            condition_data = df[df['condition'] == condition]
            
            # 各ファイルを処理
            for i, row in condition_data.iterrows():
                # サンプル番号を計算（1からスタート）
                sample_num = len(new_paths) + 1
                
                # 元のファイルパス
                source_path = row['raw_sample_path']
                file_name = os.path.basename(source_path)
                
                # 新しいディレクトリパスを作成
                new_dir = f"{setting.output_path}/{project}/span_{span}/{condition}/sample_{sample_num}/01 raw"
                file_dir = f"{setting.output_path}/{project}/span_{span}/{condition}/sample_{sample_num}"
                
                # ディレクトリを作成
                os.makedirs(new_dir, exist_ok=True)
                
                # コピー先のパス
                destination_path = os.path.join(new_dir, file_name)
                
                try:
                    # ファイルをコピー
                    shutil.copy2(source_path, destination_path)
                    print(f"Copied: {source_path} -> {destination_path}")
                except FileNotFoundError:
                    print(f"Warning: Source file not found - {source_path}")
                except Exception as e:
                    print(f"Error copying {source_path}: {str(e)}")
                
                # 新しいパスをリストに追加
                new_paths.append(file_dir)
        
        # 新しいパスをDataFrameに追加
        df['new_sample_path'] = new_paths
        
        return df

    # ファイルの振り分け
    result_df = organize_files(setting.input_csv_path, setting.project, setting.span)
    
    for _, row in result_df.iterrows():
        if row['condition'] == "condition":  # conditionが"condition"の時は除外
            continue
        print(row)
        sample = Sample(
            condition=str(row['condition']),
            sample_path=str(row['new_sample_path']),
            span=setting.span,
            magnification=float(row['magnification']),
            setting=setting
        )

    print("Finished")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time to process: {elapsed_time:.2f}sec")
