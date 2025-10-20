<div align="center">

# 🎨 DeepSeek-OCR Web UI

<img src="images/ui-preview.png" alt="DeepSeek OCR Web UI" width="100%">

### 🚀 开箱即用的智能OCR识别系统

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12+-green.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-00C7B7.svg)](https://fastapi.tiangolo.com/)
[![vLLM](https://img.shields.io/badge/vLLM-0.8.5-orange.svg)](https://github.com/vllm-project/vllm)
[![GitHub Stars](https://img.shields.io/github/stars/neosun100/DeepSeek-OCR-WebUI?style=social)](https://github.com/neosun100/DeepSeek-OCR-WebUI)

[English](README_EN.md) | [简体中文](README.md)

</div>

---

## 📖 目录

- [✨ 项目亮点](#-项目亮点)
- [🎯 功能特性](#-功能特性)
- [🖼️ 界面预览](#️-界面预览)
- [📊 识别模式](#-识别模式)
- [🚀 快速开始](#-快速开始)
- [📦 环境要求](#-环境要求)
- [⚙️ 详细安装](#️-详细安装)
- [💡 使用指南](#-使用指南)
- [🔧 高级配置](#-高级配置)
- [⚠️ 限制说明](#️-限制说明)
- [🐛 常见问题](#-常见问题)
- [📝 更新日志](#-更新日志)
- [🤝 贡献指南](#-贡献指南)
- [📄 许可证](#-许可证)
- [🙏 致谢](#-致谢)

---

## ✨ 项目亮点

> **本项目基于 [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) 官方模型，提供了一个完整的 Web UI 界面，让 OCR 识别变得前所未有的简单！**

### 🎁 为什么选择 DeepSeek-OCR Web UI？

- **🎨 现代化界面**：渐变色设计、卡片式布局、流畅动画，提供顶级用户体验
- **📱 完全响应式**：完美适配 PC、平板、手机，随时随地使用
- **🚀 开箱即用**：无需 API Key，一键启动，立即使用
- **🔄 批量处理**：支持多图片批量上传，拖拽排序，一次性识别
- **🎯 5种识别模式**：文档、OCR、纯文本、图表、图像描述，应对各种场景
- **📊 实时进度**：毫秒级日志、详细统计、进度追踪，全程透明
- **🔄 模式切换重识**：识别后可切换模式重新识别，快速对比效果
- **💾 结果管理**：一键复制、下载 TXT，结果自动合并
- **🚀 高性能**：基于 vLLM 引擎，GPU 加速，识别速度快

### 🆚 对比原项目

| 特性 | 原项目 | 本项目 (Web UI) |
|-----|-------|----------------|
| 使用方式 | 命令行 / Python代码 | 可视化 Web 界面 |
| 上手难度 | 需要编程经验 | 开箱即用，零门槛 |
| 批量处理 | 需要编写脚本 | 拖拽上传，一键处理 |
| 结果查看 | 终端输出 / 文件 | 实时显示，可复制下载 |
| 进度跟踪 | 无 | 详细日志，实时进度 |
| 模式切换 | 修改代码 | 点击按钮即可切换 |
| 移动端 | 不支持 | 完美支持 |

---

## 🎯 功能特性

### 核心功能

#### 📤 智能上传
- ✅ 拖拽上传，支持批量
- ✅ 点击上传，自动预览
- ✅ 格式验证：JPG, PNG, JPEG, BMP, GIF
- ✅ 大小统计：显示文件大小和数量

#### 🎨 图片管理
- ✅ 网格式预览，清晰直观
- ✅ 拖拽排序，调整识别顺序
- ✅ 序号显示，一目了然
- ✅ 单独删除，灵活管理

#### 🚀 识别处理
- ✅ **5种识别模式**：
  - 📄 **文档转Markdown**：保留文档格式和布局
  - 📝 **通用OCR**：提取所有可见文字
  - 📋 **纯文本提取**：纯文本不保留格式
  - 📊 **图表解析**：识别图表公式等
  - 🖼️ **图像描述**：生成详细图像描述
- ✅ **逐一处理**：按顺序识别每张图片
- ✅ **实时进度**：进度条、百分比、剩余数量
- ✅ **状态显示**：待处理、识别中、已完成、失败

#### 🔄 模式切换重识别
- ✅ **智能重置**：识别完成后切换模式，自动重置状态
- ✅ **快速对比**：用不同模式识别同一批图片，对比效果
- ✅ **无需重传**：无需刷新页面或重新上传
- ✅ **详细提示**：Toast 提示和日志记录

#### 📋 结果管理
- ✅ **自动合并**：多张图片结果合并为一个文本
- ✅ **格式化显示**：带分隔符和文件名标注
- ✅ **一键复制**：复制到剪贴板
- ✅ **下载TXT**：自动命名含日期

#### 📊 详细日志
- ✅ **毫秒级时间戳**：精确到毫秒 (HH:mm:ss.SSS)
- ✅ **彩色类型标签**：SUCCESS/ERROR/WARNING/INFO
- ✅ **三层信息结构**：主消息 + 详细描述 + 数据统计
- ✅ **完整操作追踪**：记录每一个操作细节
- ✅ **性能分析**：处理时间、字符数、进度统计
- ✅ **错误诊断**：详细错误信息和建议

---

## 🖼️ 界面预览

### 主界面
![Main Interface](images/ui-preview.png)

### 功能亮点

<table>
<tr>
<td width="50%">

#### 📱 响应式设计
- 自适应各种屏幕尺寸
- PC、平板、手机完美适配
- 流畅的动画效果

</td>
<td width="50%">

#### 🎨 现代化UI
- 渐变色设计
- 卡片式布局
- 清晰的视觉层次

</td>
</tr>
<tr>
<td width="50%">

#### 📊 实时反馈
- Toast 通知提示
- 详细日志记录
- 进度条显示

</td>
<td width="50%">

#### 🔄 流畅交互
- 拖拽排序
- 平滑过渡
- 即时响应

</td>
</tr>
</table>

---

## 📊 识别模式

### 模式详解

| 模式 | 图标 | 适用场景 | 特点 |
|-----|-----|---------|-----|
| **文档转Markdown** | 📄 | 文档、报告、论文 | 保留文档格式和布局，支持表格、列表 |
| **通用OCR** | 📝 | 各类图片文字 | 提取所有可见文字，通用性强 |
| **纯文本提取** | 📋 | 简单文字提取 | 纯文本输出，不保留格式 |
| **图表解析** | 📊 | 图表、公式、图形 | 专门识别图表、公式等复杂内容 |
| **图像描述** | 🖼️ | 图片内容理解 | 生成详细的图像描述文本 |

### 使用建议

```
📄 文档扫描件 → 文档转Markdown
📝 照片中的文字 → 通用OCR  
📋 快速提取文字 → 纯文本提取
📊 数学公式/图表 → 图表解析
🖼️ 理解图片内容 → 图像描述
```

### 模式对比测试

> 💡 **实用技巧**：上传图片后，可以依次尝试不同模式，对比识别效果，选择最佳模式！

---

## 🚀 快速开始

### 方式一：使用 Docker（推荐）

```bash
# 1. 克隆项目
git clone https://github.com/neosun100/DeepSeek-OCR-WebUI.git
cd DeepSeek-OCR-WebUI

# 2. 使用 Docker Compose 启动
docker-compose up -d

# 3. 访问 Web 界面
# 浏览器打开: http://localhost:8001
```

### 方式二：手动安装

```bash
# 1. 克隆项目
git clone https://github.com/neosun100/DeepSeek-OCR-WebUI.git
cd DeepSeek-OCR-WebUI

# 2. 创建虚拟环境
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr

# 3. 安装依赖
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install flash-attn==2.7.2.post1 --no-build-isolation

# 4. 启动服务
python web_service.py 8001

# 5. 访问 Web 界面
# 浏览器打开: http://localhost:8001
```

### 方式三：使用 Systemd（生产环境）

```bash
# 1. 复制服务文件
sudo cp deepseek-ocr.service /etc/systemd/system/

# 2. 修改服务文件中的路径
sudo nano /etc/systemd/system/deepseek-ocr.service

# 3. 启动服务
sudo systemctl daemon-reload
sudo systemctl enable deepseek-ocr
sudo systemctl start deepseek-ocr

# 4. 查看状态
sudo systemctl status deepseek-ocr

# 5. 访问 Web 界面
# 浏览器打开: http://localhost:8001
```

---

## 📦 环境要求

### 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|-----|---------|---------|
| **GPU** | NVIDIA GPU (6GB+ VRAM) | A100 40GB / RTX 4090 |
| **显存** | 8GB | 24GB+ |
| **内存** | 16GB | 32GB+ |
| **存储** | 50GB | 100GB+ SSD |
| **CUDA** | 11.8+ | 12.4+ |

### 软件要求

- **操作系统**：Linux (Ubuntu 20.04+ 推荐)
- **Python**：3.12+
- **CUDA**：11.8 / 12.4
- **PyTorch**：2.6.0
- **vLLM**：0.8.5

### 浏览器支持

- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

---

## ⚙️ 详细安装

### 1. 系统准备

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y git wget curl build-essential

# 安装 NVIDIA 驱动和 CUDA (如果尚未安装)
# 参考: https://developer.nvidia.com/cuda-downloads
```

### 2. 安装 Conda

```bash
# 下载 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 安装
bash Miniconda3-latest-Linux-x86_64.sh

# 初始化
source ~/.bashrc
```

### 3. 创建环境

```bash
# 创建虚拟环境
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr
```

### 4. 安装依赖

#### 4.1 安装 PyTorch (CUDA 11.8)

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu118
```

#### 4.2 安装 vLLM

```bash
# 下载 vLLM wheel 文件
wget https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl

# 安装
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
```

#### 4.3 安装其他依赖

```bash
# 安装项目依赖
pip install -r requirements.txt

# 安装 Flash Attention
pip install flash-attn==2.7.2.post1 --no-build-isolation
```

### 5. 下载模型

模型会在首次运行时自动从 Hugging Face 下载。

如需手动下载：

```bash
# 使用 huggingface-cli
pip install huggingface-hub

# 下载模型
huggingface-cli download deepseek-ai/DeepSeek-OCR \
    --local-dir ./models/DeepSeek-OCR
```

### 6. 启动服务

```bash
# 激活环境
conda activate deepseek-ocr

# 启动 Web 服务
python web_service.py 8001

# 或使用后台运行
nohup python web_service.py 8001 > logs/service.log 2>&1 &
```

### 7. 访问界面

打开浏览器访问：`http://localhost:8001`

---

## 💡 使用指南

### 基础使用流程

```
1. 打开 Web 界面
   ↓
2. 选择识别模式（默认：文档转Markdown）
   ↓
3. 上传图片（拖拽或点击）
   ↓
4. 调整图片顺序（可选）
   ↓
5. 点击"开始识别"
   ↓
6. 查看识别进度和日志
   ↓
7. 查看识别结果
   ↓
8. 复制或下载结果
```

### 高级使用技巧

#### 1️⃣ 批量处理多张图片

```
📁 准备图片 → 拖拽到上传区域 → 自动添加到队列 → 开始识别
```

#### 2️⃣ 调整识别顺序

```
📷 上传图片后 → 拖动图片卡片 → 调整到想要的位置 → 序号自动更新
```

#### 3️⃣ 对比不同模式效果

```
上传图片 → 用"文档转Markdown"识别 → 查看结果
         ↓
切换到"通用OCR" → 自动重置状态 → 重新识别 → 对比效果
```

#### 4️⃣ 查看详细日志

```
识别过程中 → 滚动到页面底部 → 查看"操作日志"区域
                                 ↓
                    查看处理时间、字符数、进度等详细信息
```

#### 5️⃣ 处理识别失败的图片

```
识别完成 → 查看日志中的 ERROR 记录 → 了解失败原因
           ↓
删除失败的图片 → 调整图片质量 → 重新上传 → 再次识别
```

### 使用场景示例

#### 场景1：扫描文档批量识别

```bash
目标：将10页扫描PDF转换为Markdown文本

步骤：
1. 将PDF转换为图片（每页一张）
2. 选择"文档转Markdown"模式
3. 批量上传10张图片
4. 确认图片顺序正确
5. 点击"开始识别"
6. 等待处理完成（约2-5分钟）
7. 下载合并后的Markdown文本
```

#### 场景2：照片文字快速提取

```bash
目标：提取照片中的文字内容

步骤：
1. 选择"通用OCR"模式
2. 上传照片
3. 点击"开始识别"
4. 复制识别结果
```

#### 场景3：数学公式识别

```bash
目标：识别教材中的数学公式

步骤：
1. 选择"图表解析"模式
2. 上传包含公式的图片
3. 点击"开始识别"
4. 获取LaTeX格式的公式
```

---

## 🔧 高级配置

### 修改服务端口

编辑 `web_service.py`：

```python
if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001  # 修改默认端口
    uvicorn.run(app, host="0.0.0.0", port=port)
```

或启动时指定：

```bash
python web_service.py 9000  # 使用9000端口
```

### GPU 内存优化

编辑 `web_service.py` 中的 `load_vllm_engine()` 函数：

```python
def load_vllm_engine():
    engine_args = AsyncEngineArgs(
        model="deepseek-ai/DeepSeek-OCR",
        trust_remote_code=True,
        gpu_memory_utilization=0.6,  # 调整GPU内存使用率（0.3-0.9）
        max_model_len=8192,           # 调整最大序列长度
        block_size=256,               # 调整块大小
    )
```

**建议配置**：

- **GPU < 12GB**：`gpu_memory_utilization=0.4`, `max_model_len=4096`
- **GPU 12-24GB**：`gpu_memory_utilization=0.6`, `max_model_len=8192`（默认）
- **GPU > 24GB**：`gpu_memory_utilization=0.8`, `max_model_len=16384`

### 配置 Nginx 反向代理

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # 增加超时时间（OCR处理可能较慢）
        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
    }
}
```

### 配置 HTTPS

```bash
# 使用 Certbot 获取 SSL 证书
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

## ⚠️ 限制说明

### 技术限制

| 限制项 | 说明 | 影响 |
|-------|-----|-----|
| **GPU内存** | 至少需要8GB显存 | 显存不足会导致OOM错误 |
| **并发处理** | 单个请求顺序处理 | 不支持多图片并行识别 |
| **文件大小** | 建议单张图片 < 10MB | 过大文件可能超时 |
| **图片格式** | JPG, PNG, JPEG, BMP, GIF | 其他格式需转换 |
| **批量数量** | 建议单次 < 20张 | 过多图片处理时间长 |
| **处理速度** | 约20-60秒/张 | 取决于图片复杂度和GPU |

### 功能限制

- ❌ **不支持并行识别**：图片按顺序逐一处理
- ❌ **不支持PDF直接上传**：需先转换为图片
- ❌ **不支持实时视频流**：仅支持静态图片
- ❌ **不支持多用户隔离**：无用户系统
- ❌ **不支持历史记录**：刷新页面后数据丢失

### 性能参考

| GPU型号 | 显存 | 处理速度 | 推荐配置 |
|--------|-----|---------|---------|
| RTX 3090 | 24GB | ~25秒/张 | ⭐⭐⭐⭐⭐ |
| RTX 4090 | 24GB | ~20秒/张 | ⭐⭐⭐⭐⭐ |
| A100 40GB | 40GB | ~15秒/张 | ⭐⭐⭐⭐⭐ |
| RTX 3060 | 12GB | ~40秒/张 | ⭐⭐⭐⭐ |
| GTX 1080 Ti | 11GB | ~50秒/张 | ⭐⭐⭐ |

### 使用建议

✅ **推荐做法**：
- 图片清晰度 ≥ 300 DPI
- 单次上传 5-10 张图片
- 复杂文档使用"文档转Markdown"
- 定期清空已识别的图片

❌ **不推荐做法**：
- 上传模糊或倾斜的图片
- 单次上传超过50张图片
- 在识别过程中刷新页面
- 多个浏览器标签同时识别

---

## 🐛 常见问题

### Q1: 启动时报错 "No module named 'flash_attn'"

**原因**：Flash Attention 未正确安装。

**解决方案**：

```bash
# 确认 CUDA 版本
nvcc --version

# 安装对应版本的 flash-attn
pip install flash-attn==2.7.2.post1 --no-build-isolation

# 如果安装失败，设置正确的 CUDA 路径
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
pip install flash-attn==2.7.2.post1 --no-build-isolation
```

### Q2: 识别时报错 "CUDA out of memory"

**原因**：GPU 显存不足。

**解决方案**：

1. **降低 GPU 内存使用率**（编辑 `web_service.py`）：
   ```python
   gpu_memory_utilization=0.4  # 从 0.6 降低到 0.4
   max_model_len=4096          # 从 8192 降低到 4096
   ```

2. **关闭其他 GPU 进程**：
   ```bash
   # 查看 GPU 使用情况
   nvidia-smi
   
   # 杀死占用 GPU 的进程
   kill -9 <PID>
   ```

3. **使用更小的图片**：
   - 将图片分辨率降低到 1920x1080 以下

### Q3: 无法访问 Web 界面

**检查步骤**：

1. **确认服务是否运行**：
   ```bash
   # 查看进程
   ps aux | grep web_service
   
   # 查看端口
   netstat -tulpn | grep 8001
   ```

2. **检查防火墙**：
   ```bash
   # 开放端口
   sudo ufw allow 8001
   
   # 或临时关闭防火墙
   sudo ufw disable
   ```

3. **查看服务日志**：
   ```bash
   # 如果使用 systemd
   sudo journalctl -u deepseek-ocr -f
   
   # 如果使用 nohup
   tail -f logs/service.log
   ```

### Q4: 识别结果不准确

**优化建议**：

1. **选择正确的识别模式**：
   - 文档 → 文档转Markdown
   - 图表公式 → 图表解析
   - 普通照片 → 通用OCR

2. **提高图片质量**：
   - 分辨率 ≥ 300 DPI
   - 避免模糊、倾斜
   - 保证光线充足

3. **尝试不同模式对比**：
   - 切换模式重新识别
   - 选择效果最好的结果

### Q5: 页面刷新后数据丢失

**原因**：前端数据未持久化，刷新后重置。

**解决方案**：

- 识别完成后及时下载结果
- 避免在识别过程中刷新页面
- 或自行添加 LocalStorage 持久化功能

### Q6: 模型下载慢或失败

**解决方案**：

1. **使用镜像源**：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

2. **手动下载**：
   ```bash
   # 使用 huggingface-cli
   huggingface-cli download deepseek-ai/DeepSeek-OCR \
       --local-dir ./models/DeepSeek-OCR
   ```

3. **使用代理**：
   ```bash
   export https_proxy=http://127.0.0.1:7890
   export http_proxy=http://127.0.0.1:7890
   ```

---

## 📝 更新日志

### v2.2 (2025-10-21)

#### 🎉 新增功能
- ✨ **模式切换重识别**：识别完成后切换模式可自动重置，无需重新上传
- 📊 **详细日志系统**：毫秒级时间戳、彩色类型标签、完整操作追踪
- 🔄 **智能状态管理**：切换模式自动检测并重置识别状态

#### 🐛 修复问题
- 🔧 修复拖拽排序总是移到最后的问题
- 🔧 修复识别后按钮永远灰色无法重识的问题
- 🔧 优化日志面板位置，从浮动窗口改为页面内部

#### 🎨 界面优化
- 💄 日志面板移至识别结果后面，保持主卡片宽度
- 💄 增加日志详细程度，显示处理时间、字符数、进度等
- 💄 优化 Toast 提示，更清晰的状态反馈

### v2.1 (2025-10-20)

#### 🎉 新增功能
- ✨ 详细操作日志，记录每一步操作
- 📊 实时进度追踪，显示处理时间和统计信息
- 🔄 拖拽排序功能，自由调整图片识别顺序

#### 🎨 界面优化
- 💄 完全响应式设计，完美适配移动端
- 💄 渐变色主题，现代化视觉效果
- 💄 流畅动画，提升交互体验

### v2.0 (2025-10-19)

#### 🎉 新增功能
- ✨ **完整 Web UI 界面**：开箱即用的可视化操作
- 📤 **批量上传**：支持多图片同时上传和识别
- 🎯 **5种识别模式**：文档、OCR、纯文本、图表、图像描述
- 📋 **结果管理**：一键复制、下载 TXT
- 🚀 **无需 API Key**：移除认证，直接使用

#### 🔧 技术改进
- ⚡ 基于 vLLM 0.8.5 引擎，性能优化
- 🔌 FastAPI 框架，RESTful API 设计
- 🎨 纯前端实现，无需额外依赖

### v1.0 (2025-01-XX)

- 🎉 基于 DeepSeek-OCR 官方模型
- 📝 命令行工具和 Python API
- 🚀 vLLM 和 Transformers 推理支持

---

## 🤝 贡献指南

我们欢迎所有形式的贡献！无论是报告 bug、提出功能建议，还是提交代码。

### 贡献方式

1. **🐛 报告 Bug**
   - 提交 [Issue](https://github.com/neosun100/DeepSeek-OCR-WebUI/issues)
   - 描述问题、重现步骤、环境信息

2. **💡 功能建议**
   - 提交 [Feature Request](https://github.com/neosun100/DeepSeek-OCR-WebUI/issues/new)
   - 说明功能需求和使用场景

3. **📝 改进文档**
   - Fork 项目
   - 修改文档
   - 提交 Pull Request

4. **💻 提交代码**
   ```bash
   # 1. Fork 项目
   # 2. 创建分支
   git checkout -b feature/your-feature
   
   # 3. 提交更改
   git commit -m "Add some feature"
   
   # 4. 推送到分支
   git push origin feature/your-feature
   
   # 5. 创建 Pull Request
   ```

### 开发规范

- 代码风格：遵循 PEP 8
- 提交信息：使用清晰的描述
- 测试：确保功能正常工作
- 文档：更新相关文档

---

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源。

### 关于 DeepSeek-OCR 模型

DeepSeek-OCR 模型由 [DeepSeek AI](https://www.deepseek.com/) 开发和维护。

- 模型仓库：[deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)
- 模型下载：[Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- 论文：[DeepSeek_OCR_paper.pdf](https://github.com/deepseek-ai/DeepSeek-OCR/blob/main/DeepSeek_OCR_paper.pdf)

本项目仅提供 Web UI 界面，不包含模型本身。使用时需遵守 DeepSeek-OCR 的相关协议。

---

## 🙏 致谢

### 核心依赖

- [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) - 强大的 OCR 模型
- [vLLM](https://github.com/vllm-project/vllm) - 高性能推理引擎
- [FastAPI](https://fastapi.tiangolo.com/) - 现代化 Web 框架
- [PyTorch](https://pytorch.org/) - 深度学习框架

### 灵感来源

感谢以下项目提供的灵感和参考：

- [Vary](https://github.com/Ucas-HaoranWei/Vary/)
- [GOT-OCR2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0/)
- [MinerU](https://github.com/opendatalab/MinerU)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [OneChart](https://github.com/LingyvKong/OneChart)

### 特别感谢

- 🙏 **DeepSeek AI** 团队开发的优秀 OCR 模型
- 🙏 **vLLM** 团队提供的高性能推理引擎
- 🙏 所有测试和反馈的用户

---

## 📞 联系方式

- **项目主页**：[GitHub](https://github.com/neosun100/DeepSeek-OCR-WebUI)
- **问题反馈**：[Issues](https://github.com/neosun100/DeepSeek-OCR-WebUI/issues)
- **功能建议**：[Discussions](https://github.com/neosun100/DeepSeek-OCR-WebUI/discussions)

---

## ⭐ Star History

如果这个项目对你有帮助，请给它一个 Star ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=neosun100/DeepSeek-OCR-WebUI&type=Date)](https://star-history.com/#neosun100/DeepSeek-OCR-WebUI&Date)

---

<div align="center">

### 🎉 感谢使用 DeepSeek-OCR Web UI！

**让 OCR 识别变得简单而强大**

[⬆ 回到顶部](#-deepseek-ocr-web-ui)

</div>

---

<div align="center">

Made with ❤️ by [neosun100](https://github.com/neosun100)

</div>
