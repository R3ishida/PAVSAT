import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import step1_detector
import step2_visualizer
import step3_branch_analyzer
import os
import glob
import sys
import threading
from ultralytics import YOLO


class Setting:
    def __init__(self):
        self.range = 10
        self.min_length_percentage = 0.01
        self.max_length_percentage = 0.5
        self.distance_ratio = 2
        self.variance_threshold = 200
        self.model = None
        self.input_csv_path = "input/input_sample_data.csv"
        self.output_path = "output"
        self.project = "analysis"
        self.span = 750


class SettingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("血管分析 設定画面")
        self.root.geometry("800x900")
        
        self.setting = Setting()
        self.create_widgets()
        
        # 標準出力をリダイレクト
        self.redirect_output()
        
    def create_widgets(self):
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # タイトル
        title_label = ttk.Label(main_frame, text="分析パラメータ設定", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # 各パラメータの入力欄
        row = 1
        
        # range
        ttk.Label(main_frame, text="Range (法線計算範囲):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.range_var = tk.IntVar(value=self.setting.range)
        ttk.Entry(main_frame, textvariable=self.range_var, width=30).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1
        
        # min_length_percentage
        ttk.Label(main_frame, text="最小長さ割合:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.min_length_var = tk.DoubleVar(value=self.setting.min_length_percentage)
        ttk.Entry(main_frame, textvariable=self.min_length_var, width=30).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1
        
        # max_length_percentage
        ttk.Label(main_frame, text="最大長さ割合:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.max_length_var = tk.DoubleVar(value=self.setting.max_length_percentage)
        ttk.Entry(main_frame, textvariable=self.max_length_var, width=30).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1
        
        # distance_ratio
        ttk.Label(main_frame, text="距離比率:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.distance_ratio_var = tk.IntVar(value=self.setting.distance_ratio)
        ttk.Entry(main_frame, textvariable=self.distance_ratio_var, width=30).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1
        
        # variance_threshold
        ttk.Label(main_frame, text="分散閾値:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.variance_var = tk.IntVar(value=self.setting.variance_threshold)
        ttk.Entry(main_frame, textvariable=self.variance_var, width=30).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1
        
        # model path
        ttk.Label(main_frame, text="モデルパス:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.model_path_var = tk.StringVar(value="model/best.pt")
        model_frame = ttk.Frame(main_frame)
        model_frame.grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Entry(model_frame, textvariable=self.model_path_var, width=25).pack(side=tk.LEFT)
        ttk.Button(model_frame, text="参照", command=self.browse_model).pack(side=tk.LEFT, padx=5)
        row += 1
        
        # input_csv_path
        ttk.Label(main_frame, text="入力CSVパス:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.csv_path_var = tk.StringVar(value=self.setting.input_csv_path)
        csv_frame = ttk.Frame(main_frame)
        csv_frame.grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Entry(csv_frame, textvariable=self.csv_path_var, width=25).pack(side=tk.LEFT)
        ttk.Button(csv_frame, text="参照", command=self.browse_csv).pack(side=tk.LEFT, padx=5)
        row += 1
        
        # output_path
        ttk.Label(main_frame, text="出力パス:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.output_path_var = tk.StringVar(value=self.setting.output_path)
        output_frame = ttk.Frame(main_frame)
        output_frame.grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Entry(output_frame, textvariable=self.output_path_var, width=25).pack(side=tk.LEFT)
        ttk.Button(output_frame, text="参照", command=self.browse_output).pack(side=tk.LEFT, padx=5)
        row += 1
        
        # project
        ttk.Label(main_frame, text="プロジェクト名:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.project_var = tk.StringVar(value=self.setting.project)
        ttk.Entry(main_frame, textvariable=self.project_var, width=30).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1
        
        # span
        ttk.Label(main_frame, text="Span:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.span_var = tk.IntVar(value=self.setting.span)
        ttk.Entry(main_frame, textvariable=self.span_var, width=30).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1
        
        # 実行ボタン
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="実行", command=self.run_analysis, 
                  width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="キャンセル", command=self.root.quit, 
                  width=15).pack(side=tk.LEFT, padx=5)
        
        # ステータス表示
        row += 1
        self.status_label = ttk.Label(main_frame, text="準備完了", 
                                     foreground="green")
        self.status_label.grid(row=row, column=0, columnspan=2, pady=10)
        
        # ターミナル出力表示エリア
        row += 1
        ttk.Label(main_frame, text="実行ログ:", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
        
        row += 1
        # スクロール可能なテキストエリア
        self.terminal_output = scrolledtext.ScrolledText(
            main_frame, 
            height=15, 
            width=90,
            wrap=tk.WORD,
            bg='black',
            fg='white',
            font=('Courier', 9)
        )
        self.terminal_output.grid(row=row, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # クリアボタン
        row += 1
        ttk.Button(main_frame, text="ログをクリア", 
                  command=self.clear_terminal).grid(row=row, column=0, columnspan=2, pady=5)
        
    def browse_model(self):
        filename = filedialog.askopenfilename(
            title="モデルファイルを選択",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        if filename:
            self.model_path_var.set(filename)
    
    def browse_csv(self):
        filename = filedialog.askopenfilename(
            title="CSVファイルを選択",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.csv_path_var.set(filename)
    
    def browse_output(self):
        dirname = filedialog.askdirectory(title="出力ディレクトリを選択")
        if dirname:
            self.output_path_var.set(dirname)
    
    def redirect_output(self):
        """標準出力をGUIにリダイレクト"""
        class TextRedirector:
            def __init__(self, text_widget):
                self.text_widget = text_widget
                
            def write(self, message):
                self.text_widget.insert(tk.END, message)
                self.text_widget.see(tk.END)
                self.text_widget.update()
                
            def flush(self):
                pass
        
        sys.stdout = TextRedirector(self.terminal_output)
        sys.stderr = TextRedirector(self.terminal_output)
    
    def clear_terminal(self):
        """ターミナル出力をクリア"""
        self.terminal_output.delete('1.0', tk.END)
    
    def update_setting(self):
        """GUIの値をSettingオブジェクトに反映"""
        self.setting.range = self.range_var.get()
        self.setting.min_length_percentage = self.min_length_var.get()
        self.setting.max_length_percentage = self.max_length_var.get()
        self.setting.distance_ratio = self.distance_ratio_var.get()
        self.setting.variance_threshold = self.variance_var.get()
        self.setting.model = YOLO(self.model_path_var.get())
        self.setting.input_csv_path = self.csv_path_var.get()
        self.setting.output_path = self.output_path_var.get()
        self.setting.project = self.project_var.get()
        self.setting.span = self.span_var.get()
    
    def run_analysis(self):
        """分析を別スレッドで実行"""
        # 実行ボタンを無効化
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Button) and child.cget('text') == '実行':
                        child.config(state='disabled')
        
        # 別スレッドで実行
        thread = threading.Thread(target=self.run_analysis_thread)
        thread.daemon = True
        thread.start()
    
    def run_analysis_thread(self):
        """分析の実行（スレッド内）"""
        try:
            print("=" * 60)
            print("分析を開始します...")
            print("=" * 60)
            
            self.status_label.config(text="分析を開始しています...", foreground="orange")
            
            # 設定を更新
            self.update_setting()
            print(f"\n設定:")
            print(f"  Range: {self.setting.range}")
            print(f"  Min length percentage: {self.setting.min_length_percentage}")
            print(f"  Max length percentage: {self.setting.max_length_percentage}")
            print(f"  Distance ratio: {self.setting.distance_ratio}")
            print(f"  Variance threshold: {self.setting.variance_threshold}")
            print(f"  Span: {self.setting.span}")
            print(f"  Input CSV: {self.setting.input_csv_path}")
            print(f"  Output path: {self.setting.output_path}")
            print(f"  Project: {self.setting.project}")
            print()
            
            # 分析実行
            base_dir = f'{self.setting.output_path}/{self.setting.project}/span_{self.setting.span}/*/'
            
            # Step 1
            print("\n" + "=" * 60)
            print("Step 1: 血管と分岐点の検出")
            print("=" * 60)
            self.status_label.config(text="Step 1: 血管と分岐点の検出中...", foreground="orange")
            step1_detector.main(self.setting)
            print("Step 1 完了\n")
            
            # Step 2
            sample_dirs = glob.glob(os.path.join(base_dir, "sample*"))
            print("\n" + "=" * 60)
            print("Step 2: 可視化")
            print("=" * 60)
            print(f"処理対象: {len(sample_dirs)} サンプル")
            self.status_label.config(text="Step 2: 可視化中...", foreground="orange")
            for i, sample_path in enumerate(sample_dirs, 1):
                print(f"  [{i}/{len(sample_dirs)}] {os.path.basename(sample_path)}")
                step2_visualizer.make_figure(sample_path)
            print("Step 2 完了\n")
            
            # Step 3
            print("\n" + "=" * 60)
            print("Step 3: 分岐点の分析")
            print("=" * 60)
            self.status_label.config(text="Step 3: 分岐点の分析中...", foreground="orange")
            for i, sample_path in enumerate(sample_dirs, 1):
                print(f"  [{i}/{len(sample_dirs)}] {os.path.basename(sample_path)}")
                step3_branch_analyzer.unify_branch(sample_path)
            print("Step 3 完了\n")
            
            print("\n" + "=" * 60)
            print("すべての分析が完了しました！")
            print("=" * 60)
            
            self.status_label.config(text="分析が完了しました!", foreground="green")
            self.root.after(0, lambda: messagebox.showinfo("完了", "分析が正常に完了しました。"))
            
        except Exception as e:
            error_msg = f"エラーが発生しました:\n{str(e)}"
            print("\n" + "!" * 60)
            print("エラー:")
            print(error_msg)
            print("!" * 60)
            self.status_label.config(text="エラーが発生しました", foreground="red")
            self.root.after(0, lambda: messagebox.showerror("エラー", error_msg))
        
        finally:
            # 実行ボタンを再度有効化
            for widget in self.root.winfo_children():
                if isinstance(widget, ttk.Frame):
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Button) and child.cget('text') == '実行':
                            self.root.after(0, lambda: child.config(state='normal'))


if __name__ == '__main__':
    root = tk.Tk()
    app = SettingGUI(root)
    root.mainloop()