# MOSS-TTS-Nano Docker 本地部署说明

这个仓库已经补充了一个基于 **ONNX CPU** 的本地 Docker 部署方案，目标是尽量做到：

```bash
docker compose up --build
```

即可直接启动：

- 本地 **WebUI**
- 本地 **HTTP API**
- 自动下载并缓存 ONNX 模型

## 1. 前置要求

你需要本机安装：

- Docker
- Docker Compose（新版本一般集成在 `docker compose` 命令里）

可先验证：

```bash
docker --version
docker compose version
```

## 2. 目录说明

本次新增的 Docker 相关文件：

- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`
- `DOCKER.md`

其中：

- `Dockerfile`：定义镜像构建方式
- `docker-compose.yml`：定义本地启动方式
- `.dockerignore`：减少无关文件进入构建上下文
- `DOCKER.md`：英文版 Docker 说明

> 说明：这套 Docker 配置现在是 **ONNX-only 依赖集**，不会在构建阶段安装完整的 PyTorch / torchaudio / transformers 运行栈。

## 3. 一键启动

在仓库根目录执行：

```bash
docker compose up --build
```

后台运行：

```bash
docker compose up -d --build
```

启动成功后，浏览器访问：

```text
http://127.0.0.1:18083
```

## 4. 容器默认行为

容器默认执行：

```bash
python app_onnx.py --host 0.0.0.0 --port 18083
```

也就是使用上游官方自带的 **ONNX Web Demo**，同时提供本地浏览器页面和后端接口。

另外，当前 `docker-compose.yml` 使用：

```yaml
network_mode: bridge
```

这样可以避免某些机器上因为 Docker Compose 自动创建项目网络而报错：

```text
all predefined address pools have been fully subnetted
```

## 5. 数据持久化

Compose 配置中已经挂载了以下目录：

- `./models:/app/models`
- `./generated_audio:/app/generated_audio`
- `./hf-cache:/app/.cache/huggingface`

含义如下：

### `models/`
保存 ONNX 模型文件。

如果首次启动时目录为空，应用会自动下载：

- `OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX`
- `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX`

### `generated_audio/`
保存生成出来的音频文件。

### `hf-cache/`
保存 Hugging Face 下载缓存，避免重复下载。

## 6. 首次启动会比较慢

第一次执行 `docker compose up --build` 时，通常会发生以下几件事：

1. 拉取 Python 基础镜像
2. 安装 Python 依赖
3. 启动 ONNX Runtime 服务
4. 自动下载模型文件

所以第一次启动慢是正常现象。

## 7. 常用命令

### 查看日志

```bash
docker compose logs -f
```

### 停止服务

```bash
docker compose down
```

### 重新构建并启动

```bash
docker compose up --build
```

### 删除容器但保留模型和输出文件

```bash
docker compose down
```

因为模型和输出都保存在宿主机挂载目录中，所以不会随容器删除而丢失。

## 8. 如果启动失败，优先检查这些问题

### 1）Docker 没装好

检查：

```bash
docker --version
docker compose version
```

### 2）网络无法下载 Hugging Face 模型

表现：首次启动卡在模型下载。

处理方式：

- 配置代理
- 预先手动下载模型到 `models/`

### 3）端口被占用

默认端口是：

```text
18083
```

如果端口冲突，可以修改 `docker-compose.yml` 中的端口映射。

### 4）机器性能有限

这是 **ONNX CPU** 路径，适合本地部署，但如果机器本身较弱：

- 首次加载会慢
- 长文本生成会慢
- 并发能力有限

## 9. 当前方案定位

当前 Docker 方案是：

- **本地部署优先**
- **CPU 优先**
- **ONNX 优先**
- **WebUI + API 同时可用**

如果后续你要做：

- NVIDIA GPU 加速
- `onnxruntime-gpu`
- PyTorch CUDA 推理
- 生产环境反向代理

建议后续再补：

- `Dockerfile.gpu`
- `docker-compose.gpu.yml`

## 10. 最简使用流程

```bash
docker compose up --build
```

然后打开：

```text
http://127.0.0.1:18083
```

如果页面能打开，就说明本地 WebUI 已经起来了。
