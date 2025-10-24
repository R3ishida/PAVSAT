# 基本モジュール
import os
import glob

# 数値・データ処理
import numpy as np
import pandas as pd

# 画像処理
import cv2

# グラフ描画
import matplotlib
matplotlib.use('Agg')  # GUI不要環境用
import matplotlib.pyplot as plt

# グラフ構造解析
import networkx as nx
from itertools import combinations

# クラスタリング
from sklearn.cluster import DBSCAN

def compute_strahler_order(G, root):
    order = {}
    def dfs(node, parent=None):
        children = [n for n in G.neighbors(node) if n != parent]
        if not children:
            order[node] = 1
            return 1
        child_orders = [dfs(child, node) for child in children]
        max_order = max(child_orders)
        if child_orders.count(max_order) > 1:
            order[node] = max_order + 1
        else:
            order[node] = max_order
        return order[node]
    dfs(root)
    return order


def unify_branch(sample_path):
    branch_result = f'{sample_path}/branch_result.csv'
    vessel_result = f'{sample_path}/vessel_result.csv'

    output_dir = f"{sample_path}/output/branch"
    os.makedirs(output_dir, exist_ok=True)

    image_path = f"{sample_path}/01 raw/scaled/*"  # 画像のパス
    image_files = glob.glob(image_path)  # 画像ファイルのリストを取得

    if image_files:  # ファイルが存在するか確認
        scaled_img = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
        
        if scaled_img is not None:
            print(f"Loaded image: {image_files[0]}")
        else:
            print(f"Failed to load image: {image_files[0]}")
    else:
        print("No image found.")

    image_height, image_width = scaled_img.shape  # グレースケールなので2次元

    branch_df_raw = pd.read_csv(branch_result)
    vessel_df_raw = pd.read_csv(vessel_result)

    # name から img_num を先に計算しておく
    branch_df_raw['img_num'] = branch_df_raw['name'].str.split('_').str[0].astype(int)
    vessel_df_raw['img_num'] = vessel_df_raw['name'].str.split('_').str[0].astype(int)

    # branch_id, vessel_id の最大値を取得して桁数を調べる
    branch_id_max   = branch_df_raw['branch_id'].max()
    vessel_id_max   = vessel_df_raw['vessel_id'].max()
    common_id_len   = max(len(str(branch_id_max)), len(str(vessel_id_max)))

    # img_num の最大値を取得して桁数を調べる
    branch_img_max  = branch_df_raw['img_num'].max()
    vessel_img_max  = vessel_df_raw['img_num'].max()
    common_img_len  = max(len(str(branch_img_max)), len(str(vessel_img_max)))

    def identify(df, target, connecting_list,
                id_len: int = None, img_len: int = None):
        # img_num, pattern_id の計算
        df['img_num']    = df['name'].str.split('_').str[0].astype(int)
        df['pattern_id'] = df['pattern'].str[-1].astype(int)

        # 引数で桁数が渡されていなければ、従来通りDataFrameから算出
        if id_len is None:
            id_len = len(str(df[f'{target}_id'].max()))
        if img_len is None:
            img_len = len(str(df['img_num'].max()))

        # スケール定義
        pattern_scale = 10 ** (id_len + img_len)
        img_scale     = 10 ** id_len

        # p-i-v_{target} の付番
        df[f'p-i-v_{target}'] = (
            df['pattern_id'] * pattern_scale
        + df['img_num']    * img_scale
        + df[f'{target}_id']
        )

        # connecting_list 側のIDリスト生成も同様に桁数を揃えて付番
        def make_ids(row):
            raw = row[f'{connecting_list}']
            s   = raw.strip().replace("set()", "").replace("{", "").replace("}", "")
            if not s:
                return []
            try:
                others = set(map(int, s.split(',')))
            except ValueError:
                return []
            return [
                row['pattern_id'] * pattern_scale
            + row['img_num']    * img_scale
            + o
                for o in others
            ]

        df[f'p-i-v_{connecting_list}'] = df.apply(make_ids, axis=1)
        return df

    # 先ほど計算した common_id_len, common_img_len を渡して呼び出し
    branch_df = identify(
        branch_df_raw,
        target='branch',
        connecting_list='vessels',
        id_len=common_id_len,
        img_len=common_img_len
    )

    vessel_df = identify(
        vessel_df_raw,
        target='vessel',
        connecting_list='branches',
        id_len=common_id_len,
        img_len=common_img_len
    )


    qualified_lists = []
    zero_count_lists = []
    for vessels in branch_df_raw['p-i-v_vessels']:
        q_list = []  # この行に対応するqualifiedのリスト
        zero_count = 0
        for vessel in vessels:
            matched_row = vessel_df_raw[vessel_df_raw['p-i-v_vessel'] == vessel]
            if not matched_row.empty:
                qualified = matched_row.iloc[0]['qualified']
                q_list.append(qualified)
                if qualified == 0:
                    zero_count += 1
            else:
                q_list.append(None)  # 見つからなかった場合は None などで埋める


        qualified_lists.append(q_list)
        zero_count_lists.append(zero_count)

    # 新しい列として追加
    branch_df_raw['qualified_list'] = qualified_lists
    branch_df_raw['zero_count'] = zero_count_lists
    branch_df_raw.head(100)

    branch_df_raw = branch_df_raw[branch_df_raw['qualified_list'].apply(lambda x: 0 in x)]
    branch_df_raw = branch_df_raw[branch_df_raw['zero_count'] > 0]

    branch_df = branch_df_raw[['sample','p-i-v_branch', 'pattern', 'p-i-v_vessels', 'qualified_list', 'modified_x', 'modified_y']]
    vessel_df = vessel_df_raw[['sample', 'p-i-v_vessel','thickness', 'pattern', 'p-i-v_branches', 'qualified', 'modified_x', 'modified_y', 'modified_forward_x', 'modified_forward_y', 'modified_reverse_x', 'modified_reverse_y',  'length']]

    united_vessel_df = pd.DataFrame(columns=['sample', 'p-i-v_vessel','thickness', 'pattern', 'p-i-v_branches', 'qualified', 'modified_x', 'modified_y', 'modified_forward_x', 'modified_forward_y', 'modified_reverse_x', 'modified_reverse_y',  'length'])
    p1_only_vessel_df = pd.DataFrame(columns=['sample', 'p-i-v_vessel','thickness', 'pattern', 'p-i-v_branches', 'qualified', 'modified_x', 'modified_y', 'modified_forward_x', 'modified_forward_y', 'modified_reverse_x', 'modified_reverse_y',  'length'])
    p2_only_vessel_df = pd.DataFrame(columns=['sample', 'p-i-v_vessel','thickness', 'pattern', 'p-i-v_branches', 'qualified', 'modified_x', 'modified_y', 'modified_forward_x', 'modified_forward_y', 'modified_reverse_x', 'modified_reverse_y',  'length'])

    # 'pattern01' と 'pattern02' のみを抽出し、qualified == 0 のみ
    vessel_df_p1 = vessel_df[(vessel_df['pattern'] == 'pattern01') & (vessel_df['qualified'] == 0)]
    vessel_df_p2 = vessel_df[(vessel_df['pattern'] == 'pattern02') & (vessel_df['qualified'] == 0)]

    vessel_id_mapping = {}  # ← 対応するIDの辞書を作成

    while not vessel_df_p1.empty:
        vessel_p1 = vessel_df_p1.iloc[0].copy()
        p2_pair_found = False
        for i in range(len(vessel_df_p2)):
            vessel_p2 = vessel_df_p2.iloc[i].copy()
            if (
                abs(vessel_p1['modified_x'] - vessel_p2['modified_x']) <= 50 and
                abs(vessel_p1['modified_y'] - vessel_p2['modified_y']) <= 50 and
                abs(vessel_p1['thickness'] - vessel_p2['thickness']) <= 20 and
                abs(vessel_p1['length'] - vessel_p2['length']) <= 50
            ):
                # マッチした vessel ID の対応を記録
                pattern01 = vessel_p1['p-i-v_vessel']
                pattern02 = vessel_p2['p-i-v_vessel']
                vessel_id_mapping[pattern02] = pattern01

                # 結合結果を united_vessel_df に追加
                united_vessel_df.loc[len(united_vessel_df)] = vessel_p1[['sample', 'p-i-v_vessel','thickness', 'pattern', 'p-i-v_branches', 'qualified', 'modified_x', 'modified_y', 'modified_forward_x', 'modified_forward_y', 'modified_reverse_x', 'modified_reverse_y',  'length']]
                p2_pair_found = True

                # p2 を削除
                vessel_df_p2 = vessel_df_p2.drop(vessel_df_p2.index[i])
                break
        if not p2_pair_found:
            p1_only_vessel_df.loc[len(p1_only_vessel_df)] = vessel_p1[['sample', 'p-i-v_vessel','thickness', 'pattern', 'p-i-v_branches', 'qualified', 'modified_x', 'modified_y', 'modified_forward_x', 'modified_forward_y', 'modified_reverse_x', 'modified_reverse_y',  'length']]
        
        vessel_df_p1 = vessel_df_p1.drop(vessel_df_p1.index[0])
        vessel_df_p1.reset_index(drop=True, inplace=True)

    # p2 残りを p2_only に
    p2_only_vessel_df = vessel_df_p2[['sample', 'p-i-v_vessel','thickness', 'pattern', 'p-i-v_branches', 'qualified', 'modified_x', 'modified_y', 'modified_forward_x', 'modified_forward_y', 'modified_reverse_x', 'modified_reverse_y',  'length']]

    matched_vessel_df = pd.concat([united_vessel_df, p1_only_vessel_df, p2_only_vessel_df], ignore_index=True)

    plt.figure(figsize=(24, 12), dpi=100)
    plt.imshow(scaled_img, cmap='gray', vmin=0, vmax=255, extent=[0, image_width, image_height, 0])

    for _, row in matched_vessel_df.iterrows():
        branch_list = row['p-i-v_branches']
        pattern = row['pattern']
        color = 'blue' if pattern == 'pattern01' else 'red'

        if len(branch_list) == 2:
            branch_pos_list = []
            for branch_id in branch_list:
                branch_row = branch_df[branch_df['p-i-v_branch'] == branch_id]
                if branch_row.empty:
                    continue
                x = branch_row['modified_x'].values[0]
                y = branch_row['modified_y'].values[0]
                branch_pos_list.append([x, y])
                plt.scatter(x, y, color=color, s=50, alpha=0.8, edgecolors='k')

            if len(branch_pos_list) == 2:
                plt.plot([branch_pos_list[0][0], branch_pos_list[1][0]],
                        [branch_pos_list[0][1], branch_pos_list[1][1]],
                        color=color, alpha=0.7, linewidth=2.0)

    plt.axis('off')

    # 保存先（出力ファイル名）
    plt.savefig(f'{sample_path}/output/branch/branch_pre_unified.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # Step 1: ブランチ → 親血管 の対応辞書を作成
    branch_to_vessel = {}

    for _, row in matched_vessel_df.iterrows():
        vessel_id = row['p-i-v_vessel']  # または 'p-i-v_vessel' など、列名に応じて
        for branch in row['p-i-v_branches']:
            # 最初の所属先だけ記録（複数血管に所属しない前提）
            if branch not in branch_to_vessel:
                branch_to_vessel[branch] = vessel_id

    # Step 2: 必要な行を抽出
    filtered_branch_df = branch_df[branch_df['p-i-v_branch'].isin(branch_to_vessel.keys())]

    # Step 3: modified座標とbranch_idを抽出
    valid_branch_candidates = filtered_branch_df[['p-i-v_branch', 'modified_x', 'modified_y']].copy()

    # Step 4: parent_vessel列を追加
    valid_branch_candidates['parent_vessel'] = valid_branch_candidates['p-i-v_branch'].map(branch_to_vessel)

    # Step 5: インデックスリセット
    valid_branch_candidates.reset_index(drop=True, inplace=True)

    valid_branch_candidates



    # 座標だけ抽出
    coords = valid_branch_candidates[['modified_x', 'modified_y']].values

    # DBSCANでクラスタリング
    db = DBSCAN(eps=50, min_samples=1).fit(coords)  # eps=距離のしきい値、min_samples=最小密度
    valid_branch_candidates['cluster_id'] = db.labels_

    # ① クラスタごとの重心を計算
    centroids = (
        valid_branch_candidates
        .groupby('cluster_id')[['modified_x', 'modified_y']]
        .mean()
        .reset_index()
    )

    # ② プロット準備
    plt.figure(figsize=(24, 12), dpi=100)
    plt.imshow(scaled_img, cmap='gray', vmin=0, vmax=255, extent=[0, image_width, image_height, 0])

    # ③ 各ブランチ点のプロット（元のコード）
    for _, row in matched_vessel_df.iterrows():
        for branch_id in row['p-i-v_branches']:
            branch_row = branch_df[branch_df['p-i-v_branch'] == branch_id]
            if branch_row.empty:
                continue

            x = branch_row['modified_x'].values[0]
            y = branch_row['modified_y'].values[0]
            pattern = branch_row['pattern'].values[0]
            color = 'blue' if pattern == 'pattern01' else 'red'

            plt.scatter(x, y, color=color, s=10)

    # ④ クラスタの重心をプロット（大きめで強調）
    plt.scatter(centroids['modified_x'], centroids['modified_y'], 
                color='orange', s=10, label='Centroid')

    # ⑤ 軸を非表示・凡例の追加
    plt.axis('off')
    plt.legend(loc='lower right')
    # 保存先（出力ファイル名）
    plt.savefig(f'{sample_path}/output/branch/clustered_branch.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # 1. 必要な桁数を取得（branch_idの桁数はすべて同じと仮定）
    original_len = len(str(valid_branch_candidates['p-i-v_branch'].iloc[0]))

    # 2. クラスタごとに重心を計算
    centroids = (
        valid_branch_candidates
        .groupby('cluster_id')[['modified_x', 'modified_y']]
        .mean()
        .reset_index()
    )

    # 3. 新しいbranch_idの生成（3から始まる）
    centroids['new_branch_id'] = ['3' + str(i).zfill(original_len - 1) for i in range(len(centroids))]

    # 4. 古いbranch_id → 新しいbranch_id の辞書を作成
    branch_unite_pattern_dict = {}

    for _, row in centroids.iterrows():
        cluster_id = row['cluster_id']
        new_id = int(row['new_branch_id'])
        old_ids = valid_branch_candidates[valid_branch_candidates['cluster_id'] == cluster_id]['p-i-v_branch']
        for old_id in old_ids:
            branch_unite_pattern_dict[int(old_id)] = new_id

    # 5. 新しい統合後branchのDataFrame作成
    unified_branch_df = centroids.rename(columns={
        'new_branch_id': 'p-i-v_branch',
        'modified_x': 'unified_x',
        'modified_y': 'unified_y'
    })[['p-i-v_branch', 'unified_x', 'unified_y']]

    # 必要なら int にキャスト
    unified_branch_df['p-i-v_branch'] = unified_branch_df['p-i-v_branch'].astype(int)

    # 新しい vessel_df を作成するリスト
    updated_vessel_data = []

    for _, row in matched_vessel_df.iterrows():
        original_branches = row['p-i-v_branches']
        
        # 1. マッピング辞書を使って変換
        updated_branches = [branch_unite_pattern_dict.get(branch, branch) for branch in original_branches]
        
        # 2. 重複を除去し、安定した順序に（必要であれば sorted()）
        unique_branches = sorted(set(updated_branches))
        
        # 3. 更新した内容で新しい行を作成
        new_row = row.copy()
        new_row['p-i-v_branches'] = unique_branches
        updated_vessel_data.append(new_row)

    # 4. 新たな DataFrame にまとめる
    unified_vessel_df = pd.DataFrame(updated_vessel_data).reset_index(drop=True)

    plt.figure(figsize=(24, 12), dpi=100)
    plt.imshow(scaled_img, cmap='gray', vmin=0, vmax=255, extent=[0, image_width, image_height, 0])

    # 各データポイントについて直線を描画し、thickness を表示
    for _, row in unified_vessel_df.iterrows():
        branch_list = row['p-i-v_branches']
        pattern = row['pattern']  # ← 修正: .values[0] は不要
        color = 'orange'

        if len(branch_list) == 2:
            branch_pos_list = []
            for branch_id in branch_list:
                branch_row = unified_branch_df[unified_branch_df['p-i-v_branch'] == branch_id]
                if branch_row.empty:
                    continue

                x = branch_row['unified_x'].values[0]
                y = branch_row['unified_y'].values[0]
                branch_pos_list.append([x, y])

                # 点をプロット（サイズ調整可能）
                plt.scatter(x, y, color=color, s=50, alpha=0.8, edgecolors='k')

            # 直線を描画（2点とも揃っている場合のみ）
            if len(branch_pos_list) == 2:
                plt.plot([branch_pos_list[0][0], branch_pos_list[1][0]],
                        [branch_pos_list[0][1], branch_pos_list[1][1]],
                        color=color, alpha=0.7, linewidth=2.0)

    # 軸を非表示
    plt.axis('off')
    # 保存先（出力ファイル名）
    plt.savefig(f'{sample_path}/output/branch/unified_branch.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # グラフ初期化
    G = nx.Graph()

    # 1. ノード（branch）追加
    all_branch_ids = set(b for branches in unified_vessel_df['p-i-v_branches'] for b in branches)
    G.add_nodes_from(all_branch_ids)

    # 2. エッジ（vessel）追加（太さはvesselのthickness）
    for _, row in unified_vessel_df.iterrows():
        branches = row['p-i-v_branches']
        thickness = row['thickness']
        if len(branches) < 2:
            continue
        for b1, b2 in combinations(branches, 2):
            G.add_edge(b1, b2, thickness=thickness)

    # 3. 孤立ノードを除去
    G.remove_nodes_from(list(nx.isolates(G)))

    # --- Compute Strahler order and statistics on the full graph G ---
    tree = nx.minimum_spanning_tree(G)
    root = list(tree.nodes())[0]
    strahler_order_dict = compute_strahler_order(tree, root)

    # 結果を表示
    strahler_orders = list(strahler_order_dict.values())
    mean_order = np.mean(strahler_orders)
    print(f"Strahler順序 平均: {mean_order:.2f}")

    # 1. 各vesselごとに両端のStrahler順序の平均を計算 (全 unified_vessel_df で)
    strahler_avg = []
    thickness_list = []
    for _, row in unified_vessel_df.iterrows():
        branches = row['p-i-v_branches']
        if len(branches) == 2 and all(b in strahler_order_dict for b in branches):
            avg_order = np.mean([strahler_order_dict[b] for b in branches])
            strahler_avg.append(avg_order)
            thickness_list.append(row['thickness'])

    # 2. 相関を計算
    from scipy.stats import pearsonr
    if len(strahler_avg) > 1 and len(thickness_list) > 1:
        r, p = pearsonr(strahler_avg, thickness_list)
        print(f"直径 vs 階層構造（Strahler）相関: r = {r:.2f}, p = {p:.3g}")
    else:
        r, p = np.nan, np.nan
        print("Not enough data for correlation calculation.")

    # 平均パス長とクラスタリング係数 (for G)
    if nx.is_connected(G):
        avg_path_length = nx.average_shortest_path_length(G)
    else:
        avg_path_length = np.nan
    clustering_coeff = nx.average_clustering(G)

    print(f"平均パス長: {avg_path_length:.2f}")
    print(f"クラスタリング係数: {clustering_coeff:.2f}")
    nx.write_graphml(G, f"{sample_path}/output/branch/graph.graphml")

    # --- Save network statistics to a text file ---
    result_path = os.path.join(sample_path, 'output', 'branch', 'network_stats.txt')
    try:
        with open(result_path, 'w') as f:
            f.write("Network Statistics Summary\n")
            f.write(f"Average Path Length: {avg_path_length:.2f}\n")
            f.write(f"Clustering Coefficient: {clustering_coeff:.2f}\n")
            f.write(f"Mean Strahler Order: {mean_order:.2f}\n")
            f.write(f"Pearson Correlation (Thickness vs. Strahler Order): r = {r:.2f}, p = {p:.3g}\n")
    except Exception as e:
        print(f"Failed to write network statistics: {e}")

    # 4. 最大連結成分のみを抽出して描画
    if G.number_of_nodes() == 0:
        print("No connected components found.")
        return
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        G_sub = G.subgraph(largest_cc).copy()

        # --- The following visualization steps on G_sub can remain for visualization ---
        # ここで最大連結成分のノード集合を使った vessel 抽出処理を実行
        # 1. 最大連結成分のノード（＝ブランチ）集合
        largest_cc_set = set(largest_cc)

        # 2. unified_vessel から構成ブランチがすべて largest_cc に含まれる vessel を抽出
        vessels_in_largest_cc = unified_vessel_df[
            unified_vessel_df['p-i-v_branches'].apply(lambda branches: set(branches).issubset(largest_cc_set))
        ]

        # 3. vessel_id を抽出（リストやセットで）
        vessel_ids_in_largest_cc = vessels_in_largest_cc['p-i-v_vessel'].tolist()

        # --- Visualization for G_sub as before ---
        # 5. レイアウト（kamada_kawai、重みなしでエッジ長固定）
        pos = nx.kamada_kawai_layout(G_sub, weight=None)

        # 6. thicknessを色に変換
        edges = G_sub.edges(data=True)
        thicknesses = [d['thickness'] for (u, v, d) in edges]

        norm = plt.Normalize(min(thicknesses), max(thicknesses))
        edge_colors = plt.cm.viridis(norm(thicknesses))

        # 7. 描画
        plt.figure(figsize=(18, 12), dpi=100)
        nx.draw(
            G_sub, pos,
            node_color='skyblue',    # ノードカラー
            node_size=30,            # ノードサイズ
            edge_color=edge_colors,  # エッジカラー（thickness）
            width=2.0,               # エッジ太さ
            edge_cmap=plt.cm.viridis,
            with_labels=False,
            font_size=8,
        )

        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), label='Thickness')
        plt.title("Largest Connected Component (Color=Thickness, cmap=viridis)")
        plt.axis('off')

        # 8. 保存して終了
        plt.savefig(f'{sample_path}/output/branch/largest_component_graph.png', bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # --- Strahler順序をノード色として可視化 ---
        # 描画用に全ノードの位置を unified_branch_df に基づいて設定
        pos_all = {
            row['p-i-v_branch']: (row['unified_x'], row['unified_y'])
            for _, row in unified_branch_df.iterrows()
            if row['p-i-v_branch'] in G.nodes
        }

        nx.set_node_attributes(G, strahler_order_dict, 'strahler_order')
        node_orders = [strahler_order_dict.get(node, 0) for node in G.nodes()]
        norm_node = plt.Normalize(min(node_orders), max(node_orders))
        node_colors = plt.cm.plasma(norm_node(node_orders))

        plt.figure(figsize=(18, 12), dpi=100)
        nx.draw(
            G, pos_all,
            node_color=node_colors,
            node_size=80,
            edge_color='gray',
            width=1.5,
            with_labels=False
        )
        sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm_node)
        sm.set_array([])
        plt.colorbar(sm, label='Strahler Order')
        plt.title("Branch Strahler Order Visualization")
        plt.axis('off')
        plt.savefig(f"{sample_path}/output/branch/strahler_node_plot.png", bbox_inches='tight')
        plt.close()


    # unified_branch_df から座標を辞書に
    branch_pos_dict = dict(zip(unified_branch_df['p-i-v_branch'], 
                            zip(unified_branch_df['unified_x'], unified_branch_df['unified_y'])))

    # 描画開始
    plt.figure(figsize=(24, 12), dpi=100)
    plt.imshow(scaled_img, cmap='gray', vmin=0, vmax=255, extent=[0, image_width, image_height, 0])

    # 最大連結成分に属するvesselのみを描画
    for _, row in vessels_in_largest_cc.iterrows():
        branches = row['p-i-v_branches']
        thickness = row['thickness']
        
        if len(branches) < 2:
            continue  # 線が引けない
        
        for b1, b2 in combinations(branches, 2):
            if b1 in branch_pos_dict and b2 in branch_pos_dict:
                x1, y1 = branch_pos_dict[b1]
                x2, y2 = branch_pos_dict[b2]
                plt.plot([x1, x2], [y1, y2], linewidth=2.0, color='lime', alpha=0.8)

    # 軸を非表示
    plt.axis('off')
    plt.title("Vessels from Largest Connected Component (Plotted on Image)")

    # 保存先（出力ファイル名）
    plt.savefig(f'{sample_path}/output/branch/largerst_connected_component.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()

    plt.figure(figsize=(24, 12), dpi=100)
    plt.imshow(scaled_img, cmap='gray', vmin=0, vmax=255, extent=[0, image_width, image_height, 0])

    # 最大連結成分に属するvesselのみを描画
    for _, row in vessels_in_largest_cc.iterrows():
        # 直線を描画
        plt.plot([row['modified_forward_x'], row['modified_reverse_x']],
                [row['modified_forward_y'], row['modified_reverse_y']],
                color='cyan', alpha=0.7, linewidth=1.5)
        
    # 軸を非表示
    plt.axis('off')
    plt.title("Vessels from Largest Connected Component (Plotted on Image)")

    # 保存先（出力ファイル名）
    plt.savefig(f'{sample_path}/output/branch/largerst_connected_component_thickness.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # Strahler順序をカラーで画像上に重ねて表示
    plt.figure(figsize=(24, 12), dpi=100)
    plt.imshow(scaled_img, cmap='gray', vmin=0, vmax=255, extent=[0, image_width, image_height, 0])

    # --- エッジ（血管）の描画 ---
    for _, row in unified_vessel_df.iterrows():
        branches = row['p-i-v_branches']
        if len(branches) == 2:
            coords = []
            for b in branches:
                match = unified_branch_df[unified_branch_df['p-i-v_branch'] == b]
                if not match.empty:
                    coords.append((match['unified_x'].values[0], match['unified_y'].values[0]))
            if len(coords) == 2:
                (x1, y1), (x2, y2) = coords
                plt.plot([x1, x2], [y1, y2], color='lightgray', linewidth=1.0, alpha=0.7)

    # カラーマップを定義してノードごとの色を取得
    node_orders = [strahler_order_dict.get(node, 0) for node in unified_branch_df['p-i-v_branch']]
    norm_node = plt.Normalize(min(node_orders), max(node_orders))
    node_colors = [plt.cm.plasma(norm_node(strahler_order_dict.get(branch, 0)))
                   for branch in unified_branch_df['p-i-v_branch']]

    plt.scatter(
        unified_branch_df['unified_x'],
        unified_branch_df['unified_y'],
        c=node_colors,
        s=50,
        edgecolors='k'
    )

    plt.colorbar(plt.cm.ScalarMappable(norm=norm_node, cmap='plasma'), label='Strahler Order')
    plt.title("Strahler Order Overlay on Image")
    plt.axis('off')
    plt.savefig(f"{sample_path}/output/branch/strahler_node_plot_on_image.png", bbox_inches='tight')
    plt.close()

    