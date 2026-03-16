# 快速开始指南 - Bank Card OCR System

## 🚀 快速部署（使用 Conda）

### 第一步：安装工业相机驱动和软件（必须）

#### A. 安装 Sapera LT SDK

```bash
# 1. 运行 SaperaLTSDKSetup.exe 安装程序
# 2. 按照安装向导完成安装
# 3. 默认安装路径：C:\Program Files\Teledyne DALSA\Sapera\
# 4. 安装完成后重启计算机
```

**重要提示：**
- ⚠️ 必须先安装 Sapera LT SDK 才能使用工业相机
- ⚠️ 安装完成后必须重启计算机

#### B. 安装 iNspect Express VA（相机配套软件）

```bash
# 1. 运行 iNspectExpressVA_2250_x64_Setup.exe 安装程序
# 2. 按照安装向导完成安装
# 3. 安装完成后可以使用该软件测试相机功能
```

**iNspect Express VA 用途：**
- 💾 利用测试图像去确认我们的开发需求

⚠️⚠️⚠️iNspect不能正常连接相机，因为有冲突和A步骤的SDK，我们开发的程序可以模仿iNspect的功能，从而去确定我们的开发需求

#### C. 验证安装

```bash
# 打开 "Sapera CamExpert" 工具，确认能看到相机设备



**验证步骤：**
1. 连接工业相机到电脑（网线或其他接口）
2. 打开 **Sapera CamExpert**
3. 在设备列表中找到你的相机
4. 点击连接，确认能够正常采集图像
5. 记录相机的 **Server Name**（例如：GigEVision_S1049704）

**重要提示：**
- ⚠️ 确保在 Sapera CamExpert 或 iNspect Express VA 中能看到并连接相机
- ⚠️ 记录相机的 Server Name，后续配置需要使用

### 第二步：安装 Anaconda

```bash
# 下载 Anaconda（如果未安装）
# https://www.anaconda.com/download

# 验证安装
conda --version
```

### 第三步：创建 Conda 环境

```bash
# 创建 Python 3.9 环境
conda create -n card-ocr python=3.9 -y

# 激活环境
conda activate card-ocr
```

### 第四步：安装依赖

```bash
# 进入项目目录
cd E:\Bank-Card\Card-OCR-v1

# 使用 requirements.txt 安装所有依赖
pip install -r requirements.txt
```

**说明：**
- requirements.txt 包含了所有必需的 Python 包
- 安装过程可能需要几分钟，请耐心等待
- 如果安装失败，请查看下方的问题排查

### 第五步：配置相机（必须）

编辑 `config.py` 文件，修改以下关键配置：

#### A. Sapera SDK DLL 路径
```python
# 通常不需要修改，除非安装路径不同
SAPERA_DLL_PATH = r"C:\Program Files\Teledyne DALSA\Sapera\Components\NET\Bin\DALSA.SaperaLT.SapClassBasic.dll"
```

#### B. 相机服务器名称（重要！）
```python
# 必须修改为你的相机实际名称
SERVER_NAME = "Genie_M1600_1"  # ⚠️ 替换为你的相机名称
```

**如何获取相机服务器名称：**
1. 打开 **Sapera CamExpert** 
2. 在左侧设备列表中找到你的相机
3. 查看 "Server Name" 字段（例如：Genie_M1600_1）
4. 将该名称复制到 config.py 的 SERVER_NAME 中

#### C. 资源索引config.py
```python
# 如果有多个相机，修改此值（第一个相机为 0）
RESOURCE_INDEX = 0
```

#### D. 相机默认参数（可选）
```python
CAMERA_DEFAULT_PARAMS = {
    'frame_rate_hz': 6.0,           # 帧率（Hz）
    'exposure_time_us': 66500,      # 曝光时间（微秒）
    'trigger_mode': 'Off',          # 触发模式
}
```

#### E. 用户传感器设置（可选）
```python
USER_SENSOR_SETTINGS = {
    'trigger_mode': 'internal',     # 触发模式: internal/hardware/software
    'interval_ms': 1084,            # 内部定时间隔（毫秒）
    'exposure_ms': 25.0,            # 曝光时间（毫秒）
    'brightness': 50,               # 亮度（0-100%）
    'contrast': 50,                 # 对比度（0-100%）
}
```

#### F. 对比度调整方案（可选）
```python
# 对比度调整方法：
# 'lut': 使用 LUT（会弹出 SDK 警告对话框）
# 'black_level': 使用黑电平（不会弹出警告，但只能降低对比度）
# 'software': 使用软件处理（不会弹出警告，但增加 CPU 负担）
CONTRAST_METHOD = 'lut'
```

**配置检查清单：**
- [ ] SAPERA_DLL_PATH 路径正确
- [ ] SERVER_NAME 与实际相机名称一致
- [ ] RESOURCE_INDEX 设置正确（通常为 0）

### 第六步：运行程序

```bash
# 确保 conda 环境已激活
conda activate card-ocr

# 方法一：双击 run.bat（会自动使用当前环境）

# 方法二：命令行运行
python main.py
```

---

## ⚠️ 如果遇到问题

### 问题1：conda 环境未激活
```bash
# 每次运行前确保激活环境
conda activate card-ocr

# 查看当前环境
conda info --envs
```

### 问题2：Sapera SDK 未安装或路径错误
```bash
# 1. 确认已安装 SaperaLTSDKSetup.exe
# 2. 检查安装路径是否为默认路径
# 3. 如果路径不同，修改 config.py 中的 SAPERA_DLL_PATH
# 4. 安装后必须重启计算机
```

### 问题3：相机连接失败
```bash
# 1. 打开 Sapera CamExpert 或 iNspect Express VA 确认相机可见
# 2. 检查 config.py 中的 SERVER_NAME 是否正确
# 3. 确认相机已连接并通电
# 4. 确认 Sapera SDK 已正确安装并重启过计算机
# 5. 在 iNspect Express VA 中测试相机是否能正常采集图像
```

### 问题4：pythonnet 安装失败
```bash
# 方法一：使用 conda-forge
conda install -c conda-forge pythonnet

# 方法二：使用预编译版本
pip install pythonnet --only-binary :all:

# 方法三：单独安装（跳过 requirements.txt 中的 pythonnet）
pip install opencv-python Pillow numpy pytest hypothesis
pip install pythonnet --only-binary :all:
```

### 问题5：requirements.txt 安装失败
```bash
# 如果整体安装失败，可以逐个安装核心依赖
pip install opencv-python>=4.5.0
pip install Pillow>=8.0.0
pip install numpy>=1.19.0
pip install pythonnet>=2.5.0
```

### 问题6：OpenCV 导入错误
```bash
# 重新安装 OpenCV
pip uninstall opencv-python -y
pip install opencv-python
```

---

## � 完整依赖安装命令（一键复制）

```bash
# 创建并激活环境
conda create -n card-ocr python=3.9 -y
conda activate card-ocr

# 进入项目目录
cd E:\Bank-Card\Card-OCR-v1

# 安装所有依赖
pip install -r requirements.txt
```

---

## ✅ 验证安装

运行以下命令验证：

```bash
# 激活环境
conda activate card-ocr

# 验证依赖
python -c "import cv2, PIL, numpy, clr; print('✅ 所有依赖已安装')"
```

---

## 🔄 环境管理

```bash
# 激活环境
conda activate card-ocr

# 退出环境
conda deactivate

# 查看已安装的包
conda list

# 删除环境（如果需要重新安装）
conda remove -n card-ocr --all
```

---




