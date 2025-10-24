import pandas as pd
import matplotlib.pyplot as plt
import cv2
import glob
import os

def make_figure(sample_path):
    # 画像の読み込み（グレースケールで読み込む）
    csv_path = f"{sample_path}/vessel_result.csv"
    image_path = f"{sample_path}/01 raw/scaled/*"  # 画像のパス
    image_files = glob.glob(image_path)  # 画像ファイルのリストを取得

    output_dir = f"{sample_path}/output/vessel"
    os.makedirs(output_dir, exist_ok=True)

    if image_files:  # ファイルが存在するか確認
        scaled_img = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
        
        if scaled_img is not None:
            print(f"Loaded image: {image_files[0]}")
        else:
            print(f"Failed to load image: {image_files[0]}")
    else:
        print("No image found.")


    # CSVの読み込み
    df = pd.read_csv(csv_path)

    # 'pattern01' と 'pattern02' のみを抽出し、qualified == 0 のみ
    df_pattern01 = df[(df['pattern'] == 'pattern01') & (df['qualified'] == 0)]
    df_pattern02 = df[(df['pattern'] == 'pattern02') & (df['qualified'] == 0)]

    # 画像サイズを取得
    image_height, image_width = scaled_img.shape  # グレースケールなので2次元

    # プロットの設定
    plt.figure(figsize=(24, 12))
    plt.imshow(scaled_img, cmap='gray', vmin=0, vmax=255, extent=[0, image_width, image_height, 0])
    plt.scatter(df_pattern01['modified_x'], df_pattern01['modified_y'], color='blue', alpha=0.5, label='pattern01', s=10)
    plt.scatter(df_pattern02['modified_x'], df_pattern02['modified_y'], color='red', alpha=0.5, label='pattern02', s=10)
    plt.legend()
    plt.axis('off')
    output_path = f"{sample_path}/output/vessel/modified_coordinates_plot_with_image.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    # `merge()` で `iterrows()` を高速化
    merged_df = df_pattern01.merge(df_pattern02, how='cross', suffixes=('_p1', '_p2'))

    filtered_pattern01 = merged_df[
        (abs(merged_df['modified_x_p1'] - merged_df['modified_x_p2']) <= 50) &
        (abs(merged_df['modified_y_p1'] - merged_df['modified_y_p2']) <= 50) &
        (abs(merged_df['thickness_p1'] - merged_df['thickness_p2']) <= 20)
    ]['modified_x_p1'].drop_duplicates().to_frame()

    filtered_pattern01 = df_pattern01[df_pattern01['modified_x'].isin(filtered_pattern01['modified_x_p1'])]

    filtered_pattern02 = merged_df[
        (abs(merged_df['modified_x_p1'] - merged_df['modified_x_p2']) <= 50) &
        (abs(merged_df['modified_y_p1'] - merged_df['modified_y_p2']) <= 50) &
        (abs(merged_df['thickness_p1'] - merged_df['thickness_p2']) <= 20)
    ]['modified_x_p2'].drop_duplicates().to_frame()

    remaining_pattern01 = df_pattern01[~df_pattern01['modified_x'].isin(filtered_pattern01['modified_x'])]
    remaining_pattern02 = df_pattern02[~df_pattern02['modified_x'].isin(filtered_pattern02['modified_x_p2'])]

    plt.figure(figsize=(24, 12))
    plt.imshow(scaled_img, cmap='gray', vmin=0, vmax=255, extent=[0, image_width, image_height, 0])
    plt.scatter(remaining_pattern01['modified_x'], remaining_pattern01['modified_y'], color='blue', alpha=0.5, label='pattern01', s=10)
    plt.scatter(remaining_pattern02['modified_x'], remaining_pattern02['modified_y'], color='red', alpha=0.5, label='pattern02', s=10)
    plt.scatter(filtered_pattern01['modified_x'], filtered_pattern01['modified_y'], color='orange', alpha=0.5, label='overlap', s=10)
    plt.axis('off')
    output_path = f"{sample_path}/output/vessel/filtered_modified_coordinates_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    filtered_data = filtered_pattern01.drop(columns=['pattern'])
    remaining_data = pd.concat([remaining_pattern01, remaining_pattern02]).drop(columns=['pattern'])
    final_data = pd.concat([filtered_data, remaining_data])

    final_data['thickness'] = final_data['thickness'].fillna(0).clip(upper=100)

    plt.figure(figsize=(24, 12))
    plt.imshow(scaled_img, cmap='gray', vmin=0, vmax=255, extent=[0, image_width, image_height, 0])
    scatter = plt.scatter(final_data['modified_x'], final_data['modified_y'], c=final_data['thickness'], cmap='Oranges', vmin=0, vmax=100, s=10)
    plt.colorbar(scatter, label='Thickness')
    plt.axis('off')
    output_path = f"{sample_path}/output/vessel/thickness_colored_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    plt.figure(figsize=(24, 12))
    scatter = plt.scatter(final_data['modified_x'], -final_data['modified_y'], c=final_data['thickness'], cmap='Oranges', vmin=0, vmax=100, s=100)
    plt.colorbar(scatter, label='Thickness')
    plt.axis('off')
    output_path = f"{sample_path}/output/vessel/thickness_colored_plot_without_img.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(24, 12), dpi=300)
    plt.imshow(scaled_img, cmap='gray', vmin=0, vmax=255, extent=[0, image_width, image_height, 0])
    # 各データポイントについて直線を描画し、thickness を表示
    for _, row in final_data.iterrows():
        # 直線を描画
        plt.plot([row['modified_forward_x'], row['modified_reverse_x']],
                [row['modified_forward_y'], row['modified_reverse_y']],
                color='cyan', alpha=0.7, linewidth=1.5)
    # 軸を非表示
    plt.axis('off')
    # 画像を保存（高解像度）
    output_path = f"{sample_path}/output/vessel/thickness_bar_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


    ### 変更箇所　####
    # final_dataのvessel_idを重複の内容再設定（現在は重複がある）
    # 背景画像の（modified_x, modified_y）の座標にvessel_idを表示する
    final_data['vessel_id'] = range(1, len(final_data) + 1)
    # プロットの設定
    plt.figure(figsize=(24, 12))
    plt.imshow(scaled_img, cmap='gray', vmin=0, vmax=255, extent=[0, image_width, image_height, 0])    
    # 各データポイントに vessel_id を表示
    for _, row in final_data.iterrows():
        # 直線を描画
        plt.plot([row['modified_forward_x'], row['modified_reverse_x']],
                [row['modified_forward_y'], row['modified_reverse_y']],
                color='cyan', alpha=0.7, linewidth=1.5)
    for _, row in final_data.iterrows():
        plt.text(row['modified_x'], row['modified_y'], str(row['vessel_id']), 
                 fontsize=4, color='white', ha='center', va='center')
    plt.axis('off')
    output_path = f"{sample_path}/output/vessel/vessel_id_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


    csv_output_path = f"{sample_path}/output/vessel/filtered_result.csv"
    final_data.to_csv(csv_output_path, index=False)

    # 出力パスを表示
    print(f"CSV Saved: {csv_output_path}")
