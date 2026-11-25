# Push-T Forward Closed-Loop Expert + Visual ACT

本项目在 MuJoCo / robosuite 的 Panda 单臂环境中实现 T 形物体推动、专家示范采集、视觉模仿学习和闭环评估。专家控制器先执行分阶段控制，再根据实际物理反馈进行接触感知的前向闭环修正。不需要物理搜索树、状态回滚或缓存。

ACT 学习策略使用相机图像和机器人本体状态，预测未来一段七维 OSC 动作。训练使用专家轨迹；测试时只依赖当前图像和机器人状态，不调用专家控制器，不输入 T 块真值位姿或 IoU。

## 1. 环境

在已经能够运行本项目专家生成器的 Python 环境中安装训练依赖。PyTorch 和 torchvision 应使用与本机 CUDA 或 CPU 匹配的版本；如果已经安装且可以正常导入，避免不必要的升级。

```bash
python check_setup.py
python -m pip install -r requirements-act.txt
python -c "import torch, torchvision, h5py; print(torch.__version__, torch.cuda.is_available())"
```

如果使用 Linux 无显示器环境，可按原 MuJoCo 环境的配置选择 `MUJOCO_GL=egl`（GPU 渲染）或 `MUJOCO_GL=osmesa`（CPU 渲染，系统需安装相应库）。Windows / macOS 或已有正常录像的环境无需盲目修改。ACT 默认使用 CUDA（如果可用），否则使用 CPU。

## 2. 专家算法与任务定义

- 20 Hz 控制频率；动作是 3 维 OSC 位置增量、3 维旋转增量和 1 维夹爪命令，均在 [-1,1] 范围内，不是关节目标。
- 任务成功：T 块直立且在桌面上，轮廓 IoU ≥ 0.85。默认连续 10 步满足条件，末端高度 ≥ 0.895 m 后结束。
- 默认 1000 步为防止死循环的保护上限，不是任务规定的完成时间。
- 分阶段控制首先处理初始旋转与主要平移；不能继续时安全交接给前向闭环模块。后者根据当前位姿进行接触候选选择、最多四次接触的数学前瞻、单次真实执行与局部响应更新，直到成功或达到保护退出条件。
- 专家只保存真实稳定成功的连续轨迹；失败保留诊断日志。接触模型的在线更新限于当前任务，不修改标定文件。

## 3. 采集视觉 ACT 数据（先小规模跑通）

在项目根目录执行。`--act-images` 在真实专家动作之间同步抓取两路原始 RGB 图像，**不是录制 MP4 后抽帧**。默认相机为 `birdview`、`agentview`，分辨率 224×224。它沿用原录像代码的图像上下翻转规则。图像按帧写入临时 HDF5，失败或正常异常退出会清理未完成文件。强制杀进程后可能残留 `.partial.hdf5`，这些文件不会被训练读取。

```bash
# 先尝试12个0–90°场景，单进程验证相机、控制、数据格式
python collect_expert.py --collect --per-bin 2 --act-images --workers 1 --no-video --output results/act_pilot
```

每条成功轨迹生成 `results/act_pilot/act_demos/*.hdf5` 和原 `demos/*.npz`。查看 `summary.csv` 中的 `success` 与 `act_path`。请确认图像画面含有机械臂、T 块、目标轮廓；如果相机名在你的 robosuite 安装中不同，使用 `--act-cameras` 指定实际相机名。采集与评估相机顺序、分辨率必须一致（评估直接读取 checkpoint 内的配置）。

例如查看第一条示范文件的两幅初始帧（需要 Pillow，原环境已依赖）：

```bash
python -c "import glob,h5py; from PIL import Image; p=sorted(glob.glob('results/act_pilot/act_demos/*.hdf5'))[0]; f=h5py.File(p,'r'); [Image.fromarray(f['observations/images/'+c][0]).save(c+'.png') for c in f['observations/images']]; print(p,f['action'].shape,f['observations/qpos'].shape); f.close()"
```

确认有至少 8 条成功轨迹后，运行一次小模型训练以验证完整链路（6 条训练、2 条验证，一轮训练，不代表已经学会）：

```bash
python train_act.py --data results/act_pilot/act_demos --split-file splits/pilot.json --train-count 6 --val-count 2 --epochs 1 --batch-size 2 --chunk 20 --hidden 128 --layers 2 --workers 0 --output checkpoints/pilot
python evaluate_act.py --checkpoint checkpoints/pilot/best.pt --per-bin 2 --exec-steps 5 --video-count 2 --output results/act_pilot_eval
```

如果成功轨迹少于 8 条，增加采集场景或相应减小 pilot 的训练和验证数量。评估默认涵盖 0–180° 全角度的 12 个 15° 区间，每个区间 `--per-bin` 个场景，正负角度均衡；pilot 的 `--per-bin 2` 共 24 个测试场景。视频只对指定数量的代表性案例录制。

## 4. 正式采集和 20 / 50 条示范对照实验

```bash
# 两批共120个尝试场景，注意尝试数量不等于成功数量
python collect_expert.py --collect --per-bin 10 --act-images --workers 2 --no-video --output results/act_0_90
python collect_expert.py --collect --wide-angles --per-bin 10 --act-images --workers 2 --no-video --output results/act_90_180
```

正式实验至少需要 60 条**成功**轨迹，才能留出 10 条验证数据并选取 50 条训练示范。不足时用新的 `--seed`、新的输出目录继续采集；不要重复使用相同 seed。初次建议 `--workers 1` 或 `2`，确认 offscreen 渲染与内存正常后再增加。同步图像采集会比状态采集慢，两个相机的磁盘需求也较大；建议先测实际单条文件大小。不要混合不同相机、分辨率或标定文件的示范。

两组训练共用一个固定的 episode/seed 划分文件，20 条是 50 条训练数据的子集。训练归一化参数只用当前训练子集统计，不使用验证或测试数据。所有输入样本都保留所属 episode，避免相邻帧跨集合泄漏。

```bash
python train_act.py --data results/act_0_90/act_demos results/act_90_180/act_demos --split-file splits/main.json --train-count 20 --val-count 10 --epochs 100 --chunk 50 --output checkpoints/act20
python train_act.py --data results/act_0_90/act_demos results/act_90_180/act_demos --split-file splits/main.json --train-count 50 --val-count 10 --epochs 100 --chunk 50 --output checkpoints/act50
```

默认 batch size 8，可用 `--batch-size 4` 或 `2` 降低显存占用。默认不下载预训练权重；`--pretrained` 可使用 ImageNet ResNet-18 权重（两组对照应保持一致）。100 epochs 只是起点，不保证收敛；根据 validation loss 与真实闭环成功率决定是否增加训练。相同 epochs 时 50 条数据的优化步数更多，应在实验报告中说明；如要比较相同优化步数，可按数据量调整 epochs 或 `--samples-per-episode`。

## 5. ACT 闭环仿真评估

```bash
python evaluate_act.py --checkpoint checkpoints/act20/best.pt --seed 40000 --per-bin 4 --exec-steps 10 --video-count 4 --output results/eval20
python evaluate_act.py --checkpoint checkpoints/act50/best.pt --seed 40000 --per-bin 4 --exec-steps 10 --video-count 4 --output results/eval50
```

两组使用相同的 48 个独立测试场景。评估程序禁止测试 seed 与训练/验证/预留数据 seed 重叠。也可使用 `--cases-file` 复用外部独立场景清单（`seed,angle_deg`）。每次模型根据当前相机帧及机器人状态预测 K 个动作，执行前 `--exec-steps` 个，再重新观察并预测。默认 K=50、实际执行 10 步。没有时间集成（temporal aggregation），采用滚动 action chunk 的方式闭环控制；可做执行长度对照，但应使用验证场景选参数，不要根据最终测试集调参。

策略只输入图像和机器人状态；目标位姿、IoU、upright、on_table 仅用于外部评估/终止，不作为策略特征。输出 `summary.csv` 和 `metrics.csv`，包括各 seed 与角度区间的持续成功率、最终/最佳 IoU、执行步数；`videos/` 保存选定策略的真实执行录像。成功判定与专家默认设置一致（连续 10 步 IoU ≥ 0.85、直立、在桌面上且末端抬升）。20/50 模型是独立训练，最终比较测试集成功率、平均最终 IoU 和成功轨迹步数。

## 6. 数据格式

每个成功 episode 保留原 NPZ 状态轨迹及可选 HDF5 视觉轨迹。HDF5 采用 ACT 常见目录结构，但为 Panda 的七维 OSC 动作及 T+1 状态格式设计，**不能不加适配直接交给原版 ALOHA ACT loader**。本项目提供相应 loader。

```text
episode.hdf5
├── observations/
│   ├── qpos                  (T+1, P) float32
│   └── images/
│       ├── birdview          (T+1, 224, 224, 3) uint8
│       └── agentview         (T+1, 224, 224, 3) uint8
├── action                    (T, 7) float32
└── task/
    ├── target_xy_yaw         (T+1, 3)
    ├── iou                   (T+1,)
    └── success               (T+1,)
```

`qpos` 为 7 个 Panda 关节角（由 robosuite 的 sin/cos 观测用 atan2 恢复至 [-π,π]）与夹爪关节位置拼接，通常 P=9，以实际夹爪维度为准。若环境直接提供关节角则使用相同的角度表示。图像无文字标注，RGB/HWC 存储，loader 转为 CHW。HDF5 属性中记录相机顺序、schema、控制频率、种子、初始角度、轨迹长度、稳定步数与标定哈希。

时间对齐：`image[t], qpos[t] -> action[t]`，执行后得到 `image[t+1], qpos[t+1]`。T+1 末帧用于记录终点，不作为无标签训练样本。对于长度不足 K 的动作序列，loader 零填充归一化动作并生成 padding mask；Transformer 的后验编码器与 L1 loss 忽略 padding。原 `demos/*.npz` 包含 `obs/<key>`、`actions`、`stage`、`phase`、`success`、`iou`、目标与 metadata，仍可独立使用。已有的纯状态 NPZ 不会凭空获得真实相机图像，需要重新采集或另做经过一致性验证的重放采图（本项目未实现后者）。

## 7. ACT 模型原理

本项目是独立实现的 Panda/OSC 视觉 ACT 适配版，参考原始 ACT 的 action chunking、CVAE、Transformer、L1 + KL 训练目标与零潜变量推理。不是原 ALOHA 官方仓库的逐文件移植，也不包含原版所有功能（如时间集成）。

1. ResNet-18 提取每个相机的空间特征；相机和空间位置编码区分视觉 token。
2. 训练时，CVAE 后验编码器接收当前本体状态与未来专家动作 chunk，输出潜变量分布的均值、对数方差；重参数化后得到 z。Padding mask 隔离无效动作。
3. Transformer 融合图像 token、当前机器人状态和 z，使用 K 个可学习查询输出 K×7 个归一化动作。
4. 优化目标为有效动作上的平均 L1 误差 + β KL(q(z|state, action) || N(0,I))；默认 β=10。
5. 验证和推理均不使用未来专家动作，令 z=0；评估时动作反归一化并限制在 OSC 的合法区间后真正执行。

参考：
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, 2023: https://tonyzhaozh.github.io/aloha/
- ACT 官方算法代码：https://github.com/tonyzhaozh/act （参考 `detr/models/detr_vae.py` 与 `policy.py`）
- robosuite：https://github.com/ARISE-Initiative/robosuite

## 8. 项目入口与其他使用方法

```text
push_t_expert/
├── collect_expert.py         # 专家数据采集入口
├── check_setup.py            # 原环境检查
├── train_act.py              # ACT 训练入口
├── evaluate_act.py           # ACT 闭环评估入口
├── requirements.txt
├── requirements-act.txt
├── README.md
├── config/                   # 原两份标定参数
├── act/
│   ├── __init__.py
│   ├── data.py                # 验证、划分、归一化、变长 chunk
│   └── model.py               # 视觉 CVAE / Transformer
└── src/                      # 原专家算法、环境、数据采集
    └── vision_data.py        # 同步 RGB / HDF5 记录
```

不启用 `--act-images` 时，原有状态采集和典型 MP4 录像流程保持原样。例如：

```bash
python collect_expert.py --fresh --wide-angles --per-bin 4 --video --video-count 4
python collect_expert.py --collect --per-bin 100 --workers 4 --no-video
```

`--fresh` 标记评估候选；`--collect` 标记训练候选，训练 loader 只接受后者。原有 `--max-steps`、`--seed`、`--cases-file`、`--output`、`--plan-only` 等参数继续可用。MP4 使用单进程，视觉 HDF5 与多进程可结合，但应根据渲染资源测试。

## 9. 验证范围

原前向闭环专家在此前真实 90–180° 实验中成功 23/24（95.8%），但这是专家成绩，**不是 ACT 成绩**。本项目的新增视觉采集、模型与评估逻辑通过了合成数据记录/时间对齐/划分/padding 测试、CPU 反向传播、完整一轮训练保存和模拟环境闭环评估。当前开发环境未安装 MuJoCo / robosuite，故真实相机采集、CUDA 性能及真实 ACT 成功率仍需在原项目环境验证。此项目仅包含 Push-T 任务；课程必做的 Lift 环境、示范与 ACT 训练需单独完成。
