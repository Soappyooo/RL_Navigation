# RL_Navigation
基于强化学习的机器人导航。本项目Fork自[EpMineEnv](https://github.com/DRL-CASIA/EpMineEnv).  
我们设计了一个包含辅助任务的强化学习导航框架，使用PPO算法训练模型，完成点目标视觉导航。具体请参考[实验报告](./appendix/实验报告.pdf)。
<!-- 模型结构图 -->
![model](./appendix/images/baseline_with_pose.jpg)

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

# 2 Checkpoint
给出训练好的checkpoint文件，[从OneDrive下载](https://mailsucasaccn-my.sharepoint.com/:u:/g/personal/fandongxuan24_mails_ucas_ac_cn/EbHp47b8brpIpzthzeWfSZABWClaWAvpYzJ_30BxhXycnA?e=ewSthC)（42MB）  
可以通过训练脚本中的`model.set_parameters()`方法，或评估脚本中`--model-path`参数，传入checkpoint路径。  

该checkpoint的测试结果为：
```yaml
Deterministic Action:
    Mean reward: 11.51 +/- 2.38
    Mean episode length: 47.39 +/- 92.33
    Success rate: 0.96
Non-deterministic Action:
    Mean reward: 11.71 +/- 2.07
    Mean episode length: 44.18 +/- 64.36
    Success rate: 0.99
```
注意，由于Unity环境的随机性，测试结果可能会有所不同。
<!-- 插入demo视频 -->


https://github.com/user-attachments/assets/089edcca-978f-4c24-8e32-ae8bf422be79



# 3 训练

在训练前，先调整`n_envs`参数，设置训练的环境数量。由于给定的环境存在问题，训练固定种子仍然存在随机性。
默认的训练参数应该得到与Baseline_with_pose类似的结果。  
```bash
cd EpMineEnv-main
python train_ppo_simple.py
```

# 4 评估
在评估前，前往`EpMineEnv-main/models/nav_policy.py`中，修改约222行的：
```python
# pose_projection = self.pose_projection(pose.detach() / 3)  # (batch_size, hidden_dim), avoid gradient
pose_projection = self.pose_projection(x["state"][:, -1, :].float() / 3)  # use real pose and normalize
```
将第一行取消注释，第二行注释，以确保评估时使用训练得到的pose。  
使用`evaluate.py`脚本评估训练好的模型，需要指定模型路径。  
```bash
cd EpMineEnv-main
python evaluate.py --model-path /path/to/your/checkpoint.zip
```

