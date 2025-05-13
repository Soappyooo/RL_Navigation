# RL_Navigation
基于强化学习的机器人导航。本项目Fork自[EpMineEnv](https://github.com/DRL-CASIA/EpMineEnv).

# 1 环境配置
## 1.1 方案A：使用Docker和WSL2
### 1.1.1 配置WSL2图形界面和Docker
[微软WSL2官方文档](https://learn.microsoft.com/zh-cn/windows/wsl/tutorials/gui-apps)  
[Docker文档](https://docs.docker.com/engine/install/ubuntu/)  
注意在WSL2内安装Docker而不是在Windows上安装Docker Desktop。
### 1.1.2 构建镜像
```bash
cd /path/to/this/repo
docker build -t rl_nav:latest .
```
GUI的部分参考了[WSLg的文档](https://github.com/microsoft/wslg/blob/3a4c96cbe388267faf346d19470b453a817b05ac/samples/container/Containers.md)
### 1.1.3 运行容器
```bash
docker compose up -d
```
之后可以Attach到容器中。

## 1.2 方案B：使用conda
<!-- RUN conda create -n mine_env python=3.8 -y && \
    conda activate mine_env && \
    conda install -c conda-forge libstdcxx-ng -y && \
    pip install setuptools==65.5.0 pip==21 && \
    pip install wheel==0.38.0 && \
    pip install mlagents-envs opencv-python==4.5.5.64

RUN conda activate mine_env && \
    pip install stable-baselines3==1.5.0 gym numpy torch accelerate && \
    pip install protobuf~=3.20 -->
```bash
conda create -n mine_env python=3.8 -y && \
conda activate mine_env && \
pip install setuptools==65.5.0 pip==21 && \
pip install wheel==0.38.0 && \
pip install mlagents-envs opencv-python==4.5.5.64 && \
pip install stable-baselines3==1.5.0 gym torch accelerate && \
pip install protobuf~=3.20 tensorboard ipykernel pynput
```

## 1.3 最后：下载仿真环境
参考原项目[仿真环境下载](https://github.com/DRL-CASIA/EpMineEnv#%E4%BB%BF%E7%9C%9F%E7%8E%AF%E5%A2%83%E4%B8%8B%E8%BD%BD)部分。
让drl可执行文件（drl.x86_64或drl.exe）在`EpMineEnv-main/envs/SingleAgent/MineField/`目录下。(例如，在Windows下修改下载的`MineField_Windows-0510-random`文件夹名称为`MineField`，并移动到`EpMineEnv-main/envs/SingleAgent/`)