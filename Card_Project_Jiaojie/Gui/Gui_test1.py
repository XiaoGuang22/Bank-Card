# import sys
# import json
# from PyQt5.QtWidgets import (
#     QApplication, QMainWindow, QLabel, QInputDialog, QFileDialog, QPushButton, QWidget, QVBoxLayout,
#     QHBoxLayout, QListWidget, QListWidgetItem, QMessageBox, QLineEdit, QMenu
# )
# from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QCursor
# from PyQt5.QtCore import Qt, QRect, QPoint
#
# PRESET_FIELDS = ["number", "name", "date"]
#
# class AnnotateLabel(QLabel):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.start = None
#         self.end = None
#         self.rects = []  # [(rect_on_display, field)]
#         self.temp_rect = None
#         self.selected_index = -1      # 当前高亮/选中区域
#         self.hover_index = -1         # 鼠标悬停区域
#         self.scale_ratio = 1.0
#         self.setMouseTracking(True)
#         self.current_field = None
#
#         # 拖动/缩放相关
#         self.dragging = False
#         self.resizing = False
#         self.drag_start_pos = None
#         self.drag_rect_start = None
#         self.resize_dir = None  # None, "tl", "tr", "bl", "br", "l", "r", "t", "b"
#
#     def set_current_field(self, field):
#         self.current_field = field
#
#     def edge_hit_test(self, rect, pos, tol=6):
#         """判断pos是否在rect边缘tol像素内，返回方向字符串或None"""
#         x, y = pos.x(), pos.y()
#         left, right, top, bottom = rect.left(), rect.right(), rect.top(), rect.bottom()
#         # 角
#         if abs(x-left) <= tol and abs(y-top) <= tol:
#             return "tl"
#         if abs(x-right) <= tol and abs(y-top) <= tol:
#             return "tr"
#         if abs(x-left) <= tol and abs(y-bottom) <= tol:
#             return "bl"
#         if abs(x-right) <= tol and abs(y-bottom) <= tol:
#             return "br"
#         # 边
#         if abs(x-left) <= tol and top+tol < y < bottom-tol:
#             return "l"
#         if abs(x-right) <= tol and top+tol < y < bottom-tol:
#             return "r"
#         if abs(y-top) <= tol and left+tol < x < right-tol:
#             return "t"
#         if abs(y-bottom) <= tol and left+tol < x < right-tol:
#             return "b"
#         return None
#
#     def hit_test(self, pos):
#         """返回(是否命中, 区域索引, 命中类型)"""
#         for i, (rect, field) in enumerate(self.rects):
#             dir = self.edge_hit_test(rect, pos)
#             if dir:
#                 return True, i, dir
#             if rect.contains(pos):
#                 return True, i, None
#         return False, -1, None
#
#     def limit_rect(self, rect):
#         # 限制框不超出图片
#         w, h = self.width(), self.height()
#         left = max(0, min(rect.left(), w - 1))
#         right = max(0, min(rect.right(), w - 1))
#         top = max(0, min(rect.top(), h - 1))
#         bottom = max(0, min(rect.bottom(), h - 1))
#         return QRect(QPoint(left, top), QPoint(right, bottom))
#
#     def mousePressEvent(self, event):
#         if self.pixmap() is None:
#             return
#         if event.button() == Qt.RightButton:
#             # 右键菜单：删除框
#             for i, (rect, field) in enumerate(self.rects):
#                 if rect.contains(event.pos()):
#                     menu = QMenu(self)
#                     delete_action = menu.addAction("删除该标注区域")
#                     action = menu.exec_(self.mapToGlobal(event.pos()))
#                     if action == delete_action:
#                         self.rects.pop(i)
#                         self.selected_index = -1
#                         self.hover_index = -1
#                         self.update()
#                         parent = self.window()
#                         if parent and hasattr(parent, "update_list"):
#                             parent.update_list()
#                     return
#         elif event.button() == Qt.LeftButton:
#             hit, idx, dir = self.hit_test(event.pos())
#             if hit:
#                 self.selected_index = idx
#                 if dir is not None:
#                     # 缩放
#                     self.resizing = True
#                     self.resize_dir = dir
#                     self.drag_start_pos = event.pos()
#                     self.drag_rect_start = QRect(self.rects[idx][0])
#                 else:
#                     # 拖动
#                     self.dragging = True
#                     self.drag_start_pos = event.pos()
#                     self.drag_rect_start = QRect(self.rects[idx][0])
#                 self.update()
#                 parent = self.window()
#                 if parent and hasattr(parent, "update_list"):
#                     parent.update_list()
#             else:
#                 # 开始画新框
#                 self.start = event.pos()
#                 self.end = self.start
#                 self.temp_rect = None
#                 self.update()
#
#     def mouseMoveEvent(self, event):
#         if self.pixmap() is None:
#             return
#         pos = event.pos()
#         if self.dragging and self.selected_index != -1:
#             delta = pos - self.drag_start_pos
#             rect = QRect(self.drag_rect_start)
#             rect.moveTopLeft(rect.topLeft() + delta)
#             rect = self.limit_rect(rect)
#             self.rects[self.selected_index] = (rect, self.rects[self.selected_index][1])
#             self.update()
#             return
#         if self.resizing and self.selected_index != -1:
#             rect = QRect(self.drag_rect_start)
#             dx = pos.x() - self.drag_start_pos.x()
#             dy = pos.y() - self.drag_start_pos.y()
#             if self.resize_dir == "tl":
#                 rect.setTopLeft(rect.topLeft() + QPoint(dx, dy))
#             elif self.resize_dir == "tr":
#                 rect.setTopRight(rect.topRight() + QPoint(dx, dy))
#             elif self.resize_dir == "bl":
#                 rect.setBottomLeft(rect.bottomLeft() + QPoint(dx, dy))
#             elif self.resize_dir == "br":
#                 rect.setBottomRight(rect.bottomRight() + QPoint(dx, dy))
#             elif self.resize_dir == "l":
#                 rect.setLeft(rect.left() + dx)
#             elif self.resize_dir == "r":
#                 rect.setRight(rect.right() + dx)
#             elif self.resize_dir == "t":
#                 rect.setTop(rect.top() + dy)
#             elif self.resize_dir == "b":
#                 rect.setBottom(rect.bottom() + dy)
#             rect = rect.normalized()
#             rect = self.limit_rect(rect)
#             self.rects[self.selected_index] = (rect, self.rects[self.selected_index][1])
#             self.update()
#             return
#
#         # 悬停高亮
#         hover_index = -1
#         for i, (rect, field) in enumerate(self.rects):
#             if rect.contains(pos):
#                 hover_index = i
#                 break
#         if hover_index != self.hover_index:
#             self.hover_index = hover_index
#             self.update()
#
#         # 设置鼠标形状
#         hit, idx, dir = self.hit_test(pos)
#         if hit:
#             if dir is not None:
#                 if dir in ("tl", "br"):
#                     self.setCursor(Qt.SizeFDiagCursor)
#                 elif dir in ("tr", "bl"):
#                     self.setCursor(Qt.SizeBDiagCursor)
#                 elif dir in ("l", "r"):
#                     self.setCursor(Qt.SizeHorCursor)
#                 elif dir in ("t", "b"):
#                     self.setCursor(Qt.SizeVerCursor)
#                 else:
#                     self.setCursor(Qt.SizeAllCursor)
#             else:
#                 self.setCursor(Qt.OpenHandCursor)
#         else:
#             self.setCursor(Qt.ArrowCursor)
#         # 画框时也要响应
#         if self.start:
#             self.end = event.pos()
#             self.temp_rect = QRect(self.start, self.end)
#             self.update()
#
#     def mouseReleaseEvent(self, event):
#         if self.pixmap() is None:
#             return
#         if self.dragging or self.resizing:
#             self.dragging = False
#             self.resizing = False
#             self.resize_dir = None
#             self.drag_start_pos = None
#             self.drag_rect_start = None
#             self.update()
#             parent = self.window()
#             if parent and hasattr(parent, "update_list"):
#                 parent.update_list()
#             return
#         if self.start and event.button() == Qt.LeftButton:
#             self.end = event.pos()
#             rect = QRect(self.start, self.end).normalized()
#             if rect.width() > 10 and rect.height() > 10:
#                 field = self.current_field
#                 if not field:
#                     QMessageBox.warning(self, "提示", "请先在右侧选择或输入字段名")
#                 else:
#                     self.rects.append((rect, field))
#                     self.selected_index = len(self.rects) - 1
#                     parent = self.window()
#                     if parent and hasattr(parent, "update_list"):
#                         parent.update_list()
#             self.start = None
#             self.end = None
#             self.temp_rect = None
#             self.update()
#
#     def paintEvent(self, event):
#         super().paintEvent(event)
#         if self.pixmap() is None:
#             return
#         painter = QPainter(self)
#         font = QFont()
#         font.setPointSize(12)
#         painter.setFont(font)
#         for i, (rect, field) in enumerate(self.rects):
#             # 填充色
#             if i == self.selected_index:
#                 painter.setPen(QPen(QColor(255, 0, 0), 3, Qt.SolidLine))
#                 painter.setBrush(QColor(255, 0, 0, 60))
#             elif i == self.hover_index:
#                 painter.setPen(QPen(QColor(255, 140, 0), 3, Qt.DashLine))
#                 painter.setBrush(QColor(255, 220, 0, 80))  # 更亮
#             else:
#                 painter.setPen(QPen(QColor(0, 180, 255), 3))
#                 painter.setBrush(QColor(0, 180, 255, 60))
#             painter.drawRect(rect)
#             # 区域名标签
#             text = field
#             text_rect = QRect(rect.left(), rect.top() - 28, max(80, painter.fontMetrics().width(text)+14), 24)
#             painter.setPen(Qt.NoPen)
#             painter.setBrush(QColor(255, 255, 255, 230))
#             painter.drawRect(text_rect)
#             painter.setPen(QPen(QColor(0, 0, 0)))
#             painter.drawText(text_rect, Qt.AlignCenter, text)
#             # 画手柄（可选，保留原有视觉）
#             if i == self.selected_index:
#                 size = 8
#                 points = [
#                     rect.topLeft(), rect.topRight(), rect.bottomLeft(), rect.bottomRight(),
#                     QPoint(rect.left(), rect.center().y()),
#                     QPoint(rect.right(), rect.center().y()),
#                     QPoint(rect.center().x(), rect.top()),
#                     QPoint(rect.center().x(), rect.bottom()),
#                 ]
#                 for pt in points:
#                     painter.setBrush(QColor(0, 0, 255))
#                     painter.setPen(QPen(QColor(0, 0, 255)))
#                     painter.drawRect(pt.x() - size // 2, pt.y() - size // 2, size, size)
#         # 正在画的框
#         if self.temp_rect:
#             painter.setPen(QPen(QColor(50, 200, 20), 2, Qt.DashLine))
#             painter.setBrush(QColor(50, 200, 20, 40))
#             painter.drawRect(self.temp_rect)
#
#     def get_annotations(self, scale_ratio):
#         """返回原图坐标的标注"""
#         result = []
#         for rect, field in self.rects:
#             x1 = int(rect.left() / scale_ratio)
#             y1 = int(rect.top() / scale_ratio)
#             x2 = int(rect.right() / scale_ratio)
#             y2 = int(rect.bottom() / scale_ratio)
#             result.append({"field": field, "rect": [x1, y1, x2, y2]})
#         return result
#
#     def set_selected(self, index):
#         self.selected_index = index
#         self.update()
#
#     def remove_selected(self):
#         if 0 <= self.selected_index < len(self.rects):
#             self.rects.pop(self.selected_index)
#             self.selected_index = -1
#             self.hover_index = -1
#             self.update()
#             parent = self.window()
#             if parent and hasattr(parent, "update_list"):
#                 parent.update_list()
#
#     def clear_all(self):
#         self.rects.clear()
#         self.selected_index = -1
#         self.hover_index = -1
#         self.update()
#         parent = self.window()
#         if parent and hasattr(parent, "update_list"):
#             parent.update_list()
#
# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("工业相机色带文字区域标注工具")
#         self.setStyleSheet("""
#             QWidget { font-family: 'Microsoft YaHei'; font-size: 14px; }
#             QPushButton { background: #2196f3; color: #fff; border-radius: 6px; padding: 6px 16px; }
#             QPushButton:hover { background: #1976d2; }
#             QListWidget { background: #f5f5f5; }
#         """)
#         self.label = AnnotateLabel(self)
#         self.label.setStyleSheet("background: #222;")
#         self.list_widget = QListWidget()
#         self.load_button = QPushButton("加载图片")
#         self.save_button = QPushButton("保存标注")
#         self.del_button = QPushButton("删除选中区域")
#         self.clear_button = QPushButton("清空所有区域")
#         self.load_button.clicked.connect(self.load_image)
#         self.save_button.clicked.connect(self.save_annotations)
#         self.del_button.clicked.connect(self.delete_selected)
#         self.clear_button.clicked.connect(self.clear_all)
#         self.list_widget.currentRowChanged.connect(self.label.set_selected)
#
#         # 字段区
#         self.field_list = QListWidget()
#         self.field_list.addItems(PRESET_FIELDS)
#         self.field_list.setCurrentRow(0)
#         self.label.set_current_field(PRESET_FIELDS[0])
#         self.field_list.currentTextChanged.connect(self.update_current_field)
#
#         # 当前字段提示
#         self.current_field_label = QLabel(f"当前字段：{PRESET_FIELDS[0]}")
#         self.current_field_label.setStyleSheet("color:#333;font-weight:bold;")
#
#         self.field_input = QLineEdit()
#         self.field_input.setPlaceholderText("自定义字段")
#         self.add_field_btn = QPushButton("添加字段")
#         self.add_field_btn.setMaximumWidth(80)
#         self.add_field_btn.clicked.connect(self.add_field)
#
#         field_layout = QVBoxLayout()
#         field_layout.addWidget(QLabel("字段选择："))
#         field_layout.addWidget(self.field_list)
#         field_layout.addWidget(self.current_field_label)
#         field_input_row = QHBoxLayout()
#         field_input_row.addWidget(self.field_input)
#         field_input_row.addWidget(self.add_field_btn)
#         field_layout.addLayout(field_input_row)
#         field_layout.addStretch(1)
#
#         # 主布局
#         layout = QHBoxLayout()
#         left = QVBoxLayout()
#         left.addWidget(self.load_button)
#         left.addWidget(self.save_button)
#         left.addWidget(self.del_button)
#         left.addWidget(self.clear_button)
#         left.addWidget(QLabel("区域列表："))
#         left.addWidget(self.list_widget)
#         left.addStretch(1)
#         layout.addLayout(left)
#         layout.addWidget(self.label, stretch=1)
#         layout.addLayout(field_layout)
#         widget = QWidget()
#         widget.setLayout(layout)
#         self.setCentralWidget(widget)
#         self.resize(1400, 900)
#         self.orig_pixmap = None
#         self.orig_path = None
#
#     def update_current_field(self, field):
#         self.label.set_current_field(field)
#         self.current_field_label.setText(f"当前字段：{field}")
#         # 新增：如果有选中区域，改它的字段
#         idx = self.list_widget.currentRow()
#         if 0 <= idx < len(self.label.rects):
#             rect, _ = self.label.rects[idx]
#             self.label.rects[idx] = (rect, field)
#             self.label.update()
#             self.update_list()
#
#     def add_field(self):
#         text = self.field_input.text().strip()
#         if text and self.field_list.findItems(text, Qt.MatchExactly) == []:
#             self.field_list.addItem(text)
#             self.field_list.setCurrentRow(self.field_list.count() - 1)
#             self.field_input.clear()
#             self.label.set_current_field(text)
#             self.current_field_label.setText(f"当前字段：{text}")
#
#     def load_image(self):
#         path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.bmp)")
#         if path:
#             self.orig_pixmap = QPixmap(path)
#             if self.orig_pixmap.isNull():
#                 QMessageBox.warning(self, "错误", "图片加载失败！")
#                 return
#             self.orig_path = path
#             # 支持大图自适应缩放，窗口最大显示1200x900
#             max_w, max_h = 1200, 900
#             ow, oh = self.orig_pixmap.width(), self.orig_pixmap.height()
#             scale_ratio = min(max_w / ow, max_h / oh, 1.0)
#             show_pixmap = self.orig_pixmap.scaled(int(ow * scale_ratio), int(oh * scale_ratio), Qt.KeepAspectRatio, Qt.SmoothTransformation)
#             self.label.setPixmap(show_pixmap)
#             self.label.setFixedSize(show_pixmap.size())
#             self.label.scale_ratio = scale_ratio
#             self.label.clear_all()
#
#     def save_annotations(self):
#         if self.orig_pixmap is None:
#             QMessageBox.warning(self, "提示", "请先加载图片！")
#             return
#         annotations = self.label.get_annotations(self.label.scale_ratio)
#         if not annotations:
#             QMessageBox.warning(self, "提示", "请先标注区域！")
#             return
#         path, _ = QFileDialog.getSaveFileName(self, "保存标注", "", "JSON Files (*.json)")
#         if path:
#             with open(path, "w", encoding="utf-8") as f:
#                 json.dump({
#                     "image": self.orig_path,
#                     "image_shape": [self.orig_pixmap.width(), self.orig_pixmap.height()],
#                     "regions": annotations
#                 }, f, ensure_ascii=False, indent=2)
#             QMessageBox.information(self, "保存成功", "标注已保存。")
#
#     def update_list(self):
#         self.list_widget.clear()
#         for rect, field in self.label.rects:
#             item = QListWidgetItem(f"{field} ({rect.left()},{rect.top()},{rect.right()},{rect.bottom()})")
#             self.list_widget.addItem(item)
#         # 选中同步
#         self.list_widget.setCurrentRow(self.label.selected_index)
#
#     def delete_selected(self):
#         self.label.remove_selected()
#
#     def clear_all(self):
#         self.label.clear_all()
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec_())

import sys
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QInputDialog, QFileDialog, QPushButton, QWidget, QVBoxLayout,
    QHBoxLayout, QListWidget, QListWidgetItem, QMessageBox, QLineEdit, QMenu
)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QCursor
from PyQt5.QtCore import Qt, QRect, QPoint

# 1. 字段-颜色映射，可自定义
FIELD_COLORS = {
    "number": QColor(0, 180, 255),
    "name": QColor(50, 200, 20),
    "date": QColor(255, 140, 0),
}
# 新增字段时自动分配颜色（循环使用）
COLOR_PALETTE = [
    QColor(0, 180, 255), QColor(50, 200, 20), QColor(255, 140, 0),
    QColor(255, 50, 50), QColor(200, 0, 180), QColor(180, 0, 100),
    QColor(140, 70, 255), QColor(0, 200, 150)
]
PALETTE_IDX = len(FIELD_COLORS)  # 记录下一个分配颜色的索引

PRESET_FIELDS = ["number", "name", "date"]

class AnnotateLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.start = None
        self.end = None
        self.rects = []  # [(rect_on_display, field)]
        self.temp_rect = None
        self.selected_index = -1
        self.hover_index = -1
        self.scale_ratio = 1.0
        self.setMouseTracking(True)
        self.current_field = None

        # 拖动/缩放相关
        self.dragging = False
        self.resizing = False
        self.drag_start_pos = None
        self.drag_rect_start = None
        self.resize_dir = None

    def set_current_field(self, field):
        self.current_field = field

    def edge_hit_test(self, rect, pos, tol=6):
        x, y = pos.x(), pos.y()
        left, right, top, bottom = rect.left(), rect.right(), rect.top(), rect.bottom()
        if abs(x-left) <= tol and abs(y-top) <= tol:
            return "tl"
        if abs(x-right) <= tol and abs(y-top) <= tol:
            return "tr"
        if abs(x-left) <= tol and abs(y-bottom) <= tol:
            return "bl"
        if abs(x-right) <= tol and abs(y-bottom) <= tol:
            return "br"
        if abs(x-left) <= tol and top+tol < y < bottom-tol:
            return "l"
        if abs(x-right) <= tol and top+tol < y < bottom-tol:
            return "r"
        if abs(y-top) <= tol and left+tol < x < right-tol:
            return "t"
        if abs(y-bottom) <= tol and left+tol < x < right-tol:
            return "b"
        return None

    def hit_test(self, pos):
        for i, (rect, field) in enumerate(self.rects):
            dir = self.edge_hit_test(rect, pos)
            if dir:
                return True, i, dir
            if rect.contains(pos):
                return True, i, None
        return False, -1, None

    def limit_rect(self, rect):
        w, h = self.width(), self.height()
        left = max(0, min(rect.left(), w - 1))
        right = max(0, min(rect.right(), w - 1))
        top = max(0, min(rect.top(), h - 1))
        bottom = max(0, min(rect.bottom(), h - 1))
        return QRect(QPoint(left, top), QPoint(right, bottom))

    def mousePressEvent(self, event):
        if self.pixmap() is None:
            return
        if event.button() == Qt.RightButton:
            for i, (rect, field) in enumerate(self.rects):
                if rect.contains(event.pos()):
                    menu = QMenu(self)
                    delete_action = menu.addAction("删除该标注区域")
                    action = menu.exec_(self.mapToGlobal(event.pos()))
                    if action == delete_action:
                        self.rects.pop(i)
                        self.selected_index = -1
                        self.hover_index = -1
                        self.update()
                        parent = self.window()
                        if parent and hasattr(parent, "update_list"):
                            parent.update_list()
                    return
        elif event.button() == Qt.LeftButton:
            hit, idx, dir = self.hit_test(event.pos())
            if hit:
                self.selected_index = idx
                if dir is not None:
                    self.resizing = True
                    self.resize_dir = dir
                    self.drag_start_pos = event.pos()
                    self.drag_rect_start = QRect(self.rects[idx][0])
                else:
                    self.dragging = True
                    self.drag_start_pos = event.pos()
                    self.drag_rect_start = QRect(self.rects[idx][0])
                self.update()
                parent = self.window()
                if parent and hasattr(parent, "update_list"):
                    parent.update_list()
            else:
                self.start = event.pos()
                self.end = self.start
                self.temp_rect = None
                self.update()

    def mouseMoveEvent(self, event):
        if self.pixmap() is None:
            return
        pos = event.pos()
        if self.dragging and self.selected_index != -1:
            delta = pos - self.drag_start_pos
            rect = QRect(self.drag_rect_start)
            rect.moveTopLeft(rect.topLeft() + delta)
            rect = self.limit_rect(rect)
            self.rects[self.selected_index] = (rect, self.rects[self.selected_index][1])
            self.update()
            return
        if self.resizing and self.selected_index != -1:
            rect = QRect(self.drag_rect_start)
            dx = pos.x() - self.drag_start_pos.x()
            dy = pos.y() - self.drag_start_pos.y()
            if self.resize_dir == "tl":
                rect.setTopLeft(rect.topLeft() + QPoint(dx, dy))
            elif self.resize_dir == "tr":
                rect.setTopRight(rect.topRight() + QPoint(dx, dy))
            elif self.resize_dir == "bl":
                rect.setBottomLeft(rect.bottomLeft() + QPoint(dx, dy))
            elif self.resize_dir == "br":
                rect.setBottomRight(rect.bottomRight() + QPoint(dx, dy))
            elif self.resize_dir == "l":
                rect.setLeft(rect.left() + dx)
            elif self.resize_dir == "r":
                rect.setRight(rect.right() + dx)
            elif self.resize_dir == "t":
                rect.setTop(rect.top() + dy)
            elif self.resize_dir == "b":
                rect.setBottom(rect.bottom() + dy)
            rect = rect.normalized()
            rect = self.limit_rect(rect)
            self.rects[self.selected_index] = (rect, self.rects[self.selected_index][1])
            self.update()
            return

        hover_index = -1
        for i, (rect, field) in enumerate(self.rects):
            if rect.contains(pos):
                hover_index = i
                break
        if hover_index != self.hover_index:
            self.hover_index = hover_index
            self.update()

        hit, idx, dir = self.hit_test(pos)
        if hit:
            if dir is not None:
                if dir in ("tl", "br"):
                    self.setCursor(Qt.SizeFDiagCursor)
                elif dir in ("tr", "bl"):
                    self.setCursor(Qt.SizeBDiagCursor)
                elif dir in ("l", "r"):
                    self.setCursor(Qt.SizeHorCursor)
                elif dir in ("t", "b"):
                    self.setCursor(Qt.SizeVerCursor)
                else:
                    self.setCursor(Qt.SizeAllCursor)
            else:
                self.setCursor(Qt.OpenHandCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        if self.start:
            self.end = event.pos()
            self.temp_rect = QRect(self.start, self.end)
            self.update()

    def mouseReleaseEvent(self, event):
        if self.pixmap() is None:
            return
        if self.dragging or self.resizing:
            self.dragging = False
            self.resizing = False
            self.resize_dir = None
            self.drag_start_pos = None
            self.drag_rect_start = None
            self.update()
            parent = self.window()
            if parent and hasattr(parent, "update_list"):
                parent.update_list()
            return
        if self.start and event.button() == Qt.LeftButton:
            self.end = event.pos()
            rect = QRect(self.start, self.end).normalized()
            if rect.width() > 10 and rect.height() > 10:
                field = self.current_field
                if not field:
                    QMessageBox.warning(self, "提示", "请先在右侧选择或输入字段名")
                else:
                    self.rects.append((rect, field))
                    self.selected_index = len(self.rects) - 1
                    parent = self.window()
                    if parent and hasattr(parent, "update_list"):
                        parent.update_list()
            self.start = None
            self.end = None
            self.temp_rect = None
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.pixmap() is None:
            return
        painter = QPainter(self)
        font = QFont()
        font.setPointSize(12)
        painter.setFont(font)
        for i, (rect, field) in enumerate(self.rects):
            # 2. 获取字段颜色
            color = FIELD_COLORS.get(field, QColor(0, 180, 255))
            # 填充色
            if i == self.selected_index:
                painter.setPen(QPen(color, 3, Qt.SolidLine))
                painter.setBrush(QColor(color.red(), color.green(), color.blue(), 60))
            elif i == self.hover_index:
                painter.setPen(QPen(color, 3, Qt.DashLine))
                painter.setBrush(QColor(color.red(), color.green(), color.blue(), 80))
            else:
                painter.setPen(QPen(color, 3))
                painter.setBrush(QColor(color.red(), color.green(), color.blue(), 60))
            painter.drawRect(rect)
            # 区域名标签
            text = field
            text_rect = QRect(rect.left(), rect.top() - 28, max(80, painter.fontMetrics().width(text)+14), 24)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255, 230))
            painter.drawRect(text_rect)
            painter.setPen(QPen(QColor(0, 0, 0)))
            painter.drawText(text_rect, Qt.AlignCenter, text)
            # 画手柄
            if i == self.selected_index:
                size = 8
                points = [
                    rect.topLeft(), rect.topRight(), rect.bottomLeft(), rect.bottomRight(),
                    QPoint(rect.left(), rect.center().y()),
                    QPoint(rect.right(), rect.center().y()),
                    QPoint(rect.center().x(), rect.top()),
                    QPoint(rect.center().x(), rect.bottom()),
                ]
                for pt in points:
                    painter.setBrush(color)
                    painter.setPen(QPen(color))
                    painter.drawRect(pt.x() - size // 2, pt.y() - size // 2, size, size)
        if self.temp_rect:
            painter.setPen(QPen(QColor(50, 200, 20), 2, Qt.DashLine))
            painter.setBrush(QColor(50, 200, 20, 40))
            painter.drawRect(self.temp_rect)

    def get_annotations(self, scale_ratio):
        result = []
        for rect, field in self.rects:
            x1 = int(rect.left() / scale_ratio)
            y1 = int(rect.top() / scale_ratio)
            x2 = int(rect.right() / scale_ratio)
            y2 = int(rect.bottom() / scale_ratio)
            result.append({"field": field, "rect": [x1, y1, x2, y2]})
        return result

    def set_selected(self, index):
        self.selected_index = index
        self.update()

    def remove_selected(self):
        if 0 <= self.selected_index < len(self.rects):
            self.rects.pop(self.selected_index)
            self.selected_index = -1
            self.hover_index = -1
            self.update()
            parent = self.window()
            if parent and hasattr(parent, "update_list"):
                parent.update_list()

    def clear_all(self):
        self.rects.clear()
        self.selected_index = -1
        self.hover_index = -1
        self.update()
        parent = self.window()
        if parent and hasattr(parent, "update_list"):
            parent.update_list()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("工业相机色带文字区域标注工具")
        self.setStyleSheet("""
            QWidget { font-family: 'Microsoft YaHei'; font-size: 14px; }
            QPushButton { background: #2196f3; color: #fff; border-radius: 6px; padding: 6px 16px; }
            QPushButton:hover { background: #1976d2; }
            QListWidget { background: #f5f5f5; }
        """)
        self.label = AnnotateLabel(self)
        self.label.setStyleSheet("background: #222;")
        self.list_widget = QListWidget()
        self.load_button = QPushButton("加载图片")
        self.save_button = QPushButton("保存标注")
        self.del_button = QPushButton("删除选中区域")
        self.clear_button = QPushButton("清空所有区域")
        self.load_button.clicked.connect(self.load_image)
        self.save_button.clicked.connect(self.save_annotations)
        self.del_button.clicked.connect(self.delete_selected)
        self.clear_button.clicked.connect(self.clear_all)
        self.list_widget.currentRowChanged.connect(self.label.set_selected)

        # 字段区
        self.field_list = QListWidget()
        self.field_list.addItems(PRESET_FIELDS)
        self.field_list.setCurrentRow(0)
        self.label.set_current_field(PRESET_FIELDS[0])
        self.field_list.currentTextChanged.connect(self.update_current_field)

        self.current_field_label = QLabel(f"当前字段：{PRESET_FIELDS[0]}")
        self.current_field_label.setStyleSheet("color:#333;font-weight:bold;")

        self.field_input = QLineEdit()
        self.field_input.setPlaceholderText("自定义字段")
        self.add_field_btn = QPushButton("添加字段")
        self.add_field_btn.setMaximumWidth(80)
        self.add_field_btn.clicked.connect(self.add_field)

        field_layout = QVBoxLayout()
        field_layout.addWidget(QLabel("字段选择："))
        field_layout.addWidget(self.field_list)
        field_layout.addWidget(self.current_field_label)
        field_input_row = QHBoxLayout()
        field_input_row.addWidget(self.field_input)
        field_input_row.addWidget(self.add_field_btn)
        field_layout.addLayout(field_input_row)
        field_layout.addStretch(1)

        layout = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(self.load_button)
        left.addWidget(self.save_button)
        left.addWidget(self.del_button)
        left.addWidget(self.clear_button)
        left.addWidget(QLabel("区域列表："))
        left.addWidget(self.list_widget)
        left.addStretch(1)
        layout.addLayout(left)
        layout.addWidget(self.label, stretch=1)
        layout.addLayout(field_layout)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.resize(1400, 900)
        self.orig_pixmap = None
        self.orig_path = None

    def update_current_field(self, field):
        self.label.set_current_field(field)
        self.current_field_label.setText(f"当前字段：{field}")
        idx = self.list_widget.currentRow()
        if 0 <= idx < len(self.label.rects):
            rect, _ = self.label.rects[idx]
            self.label.rects[idx] = (rect, field)
            self.label.update()
            self.update_list()

    def add_field(self):
        global PALETTE_IDX
        text = self.field_input.text().strip()
        if text and self.field_list.findItems(text, Qt.MatchExactly) == []:
            self.field_list.addItem(text)
            self.field_list.setCurrentRow(self.field_list.count() - 1)
            self.field_input.clear()
            self.label.set_current_field(text)
            self.current_field_label.setText(f"当前字段：{text}")
            # 3. 新增字段自动分配颜色
            if text not in FIELD_COLORS:
                FIELD_COLORS[text] = COLOR_PALETTE[PALETTE_IDX % len(COLOR_PALETTE)]
                PALETTE_IDX += 1

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.bmp)")
        if path:
            self.orig_pixmap = QPixmap(path)
            if self.orig_pixmap.isNull():
                QMessageBox.warning(self, "错误", "图片加载失败！")
                return
            self.orig_path = path
            max_w, max_h = 1200, 900
            ow, oh = self.orig_pixmap.width(), self.orig_pixmap.height()
            scale_ratio = min(max_w / ow, max_h / oh, 1.0)
            show_pixmap = self.orig_pixmap.scaled(int(ow * scale_ratio), int(oh * scale_ratio), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label.setPixmap(show_pixmap)
            self.label.setFixedSize(show_pixmap.size())
            self.label.scale_ratio = scale_ratio
            self.label.clear_all()

    def save_annotations(self):
        if self.orig_pixmap is None:
            QMessageBox.warning(self, "提示", "请先加载图片！")
            return
        annotations = self.label.get_annotations(self.label.scale_ratio)
        if not annotations:
            QMessageBox.warning(self, "提示", "请先标注区域！")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存标注", "", "JSON Files (*.json)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "image": self.orig_path,
                    "image_shape": [self.orig_pixmap.width(), self.orig_pixmap.height()],
                    "regions": annotations
                }, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "保存成功", "标注已保存。")

    def update_list(self):
        self.list_widget.clear()
        for rect, field in self.label.rects:
            item = QListWidgetItem(f"{field} ({rect.left()},{rect.top()},{rect.right()},{rect.bottom()})")
            self.list_widget.addItem(item)
        self.list_widget.setCurrentRow(self.label.selected_index)

    def delete_selected(self):
        self.label.remove_selected()

    def clear_all(self):
        self.label.clear_all()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

