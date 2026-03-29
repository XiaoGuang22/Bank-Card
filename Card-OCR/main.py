#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bank Card OCR System - 主程序入口
银行卡识别系统 - 主程序入口

这是系统的统一启动入口，会启动主界面 InspectMainWindow。

使用方法:
    python main.py

作者: XiaoGuang
最后更新: 2026-01-23
"""

import sys
import os

# 添加当前目录到 Python 路径，确保可以导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 导入主窗口
from InspectMainWindow import InspectMainWindow
import tkinter as tk


# def main():
#     """主函数：创建并运行主窗口"""
#     print("="*60)
#     print("Bank Card OCR System - 银行卡识别系统")
#     print("="*60)
#     print("正在启动主界面...")
#     print()
#  =========替换代码======================
def login_success(username, role):
    """登录成功后的回调函数"""
    print(f"✅ 登录成功: {username} ({role})")

# =============替换结束======================
    
    # 创建主窗口
    root = tk.Tk()
    #  app = InspectMainWindow(root)
    app = InspectMainWindow(root, username, role)
    
    # 设置关闭事件处理
    def on_closing():
        """关闭程序时的清理工作"""
        print("\n正在关闭程序...")
        try:
            # 停止视频循环
            if hasattr(app, 'video_loop_running'):
                app.video_loop_running = False
            
            # 清理相机资源
            if hasattr(app, 'cam') and app.cam:
                app.cam.cleanup()
        except Exception as e:
            # 静默处理异常
            pass
        finally:
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # 启动主循环
    root.mainloop()


#   ============新增登录窗口==================
#   
def main():
    """主函数：创建并运行主窗口"""
    print("="*60)
    print("Bank Card OCR System - 银行卡识别系统")
    print("="*60)
    print("正在启动登录界面...")
    print()
    
    # 导入登录窗口
    from ui.LoginWindow import LoginWindow
    
    # 创建登录窗口
    root = tk.Tk()
    login_window = LoginWindow(root, login_success)
    
    # 启动登录窗口主循环
    root.mainloop()
# ===============新增结束==============================


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)