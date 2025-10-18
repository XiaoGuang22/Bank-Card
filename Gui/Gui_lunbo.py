import os
import json
from tkinter import Tk, Canvas, Button, NW, Label, filedialog, messagebox, Frame, CENTER
from PIL import Image, ImageTk

target_width, target_height = 800, 600

class ImageViewer:
    def __init__(self, master):
        self.master = master
        self.img_list = []
        self.img_dir = ""
        self.json_path = ""
        self.regions = []
        self.orig_width = 1600
        self.orig_height = 1200
        self.idx = 0
        self.auto_play = False

        # 文件名标签
        self.label = Label(master, text="未选择图片", font=("微软雅黑", 16, "bold"))
        self.label.pack(pady=(20, 10))

        # 图片画布
        self.canvas = Canvas(master, width=target_width, height=target_height, bg="#f7f7f7", highlightthickness=1, highlightbackground="#cccccc")
        self.canvas.pack(pady=(0, 20))

        # 按钮区
        btn_frame = Frame(master)
        btn_frame.pack(pady=(0, 20))

        btn_style = {"width": 14, "height": 2, "font": ("微软雅黑", 11)}

        self.prev_btn = Button(btn_frame, text="上一张", command=self.prev_img, **btn_style)
        self.prev_btn.pack(side='left', padx=8)
        self.next_btn = Button(btn_frame, text="下一张", command=self.next_img, **btn_style)
        self.next_btn.pack(side='left', padx=8)
        self.load_json_btn = Button(btn_frame, text="选择模板", command=self.load_json, **btn_style)
        self.load_json_btn.pack(side='left', padx=8)
        self.load_img_btn = Button(btn_frame, text="选择图片文件夹", command=self.load_img_dir, **btn_style)
        self.load_img_btn.pack(side='left', padx=8)
        self.auto_btn = Button(btn_frame, text="开始自动播放", command=self.toggle_auto_play, **btn_style)
        self.auto_btn.pack(side='left', padx=8)

        self.show_img()

    def load_json(self):
        path = filedialog.askopenfilename(title="选择json文件", filetypes=[("JSON files", "*.json")])
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    label_data = json.load(f)
                self.regions = label_data['regions']
                self.orig_width, self.orig_height = label_data['image_shape']
                self.json_path = path
                messagebox.showinfo("成功", f"已加载模板: {os.path.basename(path)}")
                self.show_img()
            except Exception as e:
                messagebox.showerror("错误", f"加载json失败: {e}")

    def load_img_dir(self):
        path = filedialog.askdirectory(title="选择图片文件夹")
        if path:
            self.img_dir = path
            self.img_list = [f for f in os.listdir(self.img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            self.idx = 0
            if not self.img_list:
                messagebox.showwarning("警告", "该文件夹下没有图片")
            self.show_img()

    def show_img(self):
        self.canvas.delete("all")
        if not self.img_list or not self.img_dir:
            self.label.config(text="未选择图片")
            return
        img_name = self.img_list[self.idx]
        self.label.config(text=img_name)
        img_path = os.path.join(self.img_dir, img_name)
        try:
            pil_img = Image.open(img_path).resize((target_width, target_height))
            self.tk_img = ImageTk.PhotoImage(pil_img)
            self.canvas.create_image(0, 0, anchor=NW, image=self.tk_img)
        except Exception as e:
            self.label.config(text=f"图片加载失败: {e}")
            return

        # 画标注
        if self.regions:
            colors = ["red", "green", "blue", "orange", "purple"]
            for idx, region in enumerate(self.regions):
                rect = region['rect']
                field = region['field']
                color = colors[idx % len(colors)]
                x1, y1, x2, y2 = rect
                x_scale = target_width / self.orig_width
                y_scale = target_height / self.orig_height
                x1, x2 = x1 * x_scale, x2 * x_scale
                y1, y2 = y1 * y_scale, y2 * y_scale
                self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
                self.canvas.create_text(x1+5, y1+15, anchor=NW, text=field, fill=color, font=("微软雅黑", 11, "bold"))

    def prev_img(self):
        if not self.img_list: return
        self.idx = (self.idx - 1) % len(self.img_list)
        self.show_img()

    def next_img(self):
        if not self.img_list: return
        self.idx = (self.idx + 1) % len(self.img_list)
        self.show_img()

    def toggle_auto_play(self):
        self.auto_play = not self.auto_play
        self.auto_btn.config(text="停止自动播放" if self.auto_play else "开始自动播放")
        if self.auto_play:
            self.auto_next_img()

    def auto_next_img(self):
        if self.auto_play and self.img_list:
            self.next_img()
            self.master.after(2000, self.auto_next_img)  # 5000毫秒=5秒

root = Tk()
root.title("图片轮播与标注显示")
root.resizable(False, False)  # 禁止缩放窗口
root.geometry(f"{target_width+60}x{target_height+180}")  # 适当加宽高
viewer = ImageViewer(root)
root.mainloop()
