# Push-T Forward Closed-Loop Expert

面向模仿学习的 T 形物体推动专家轨迹生成器。使用 MuJoCo / robosuite 中的 Panda 机械臂，通过分阶段控制、接触感知多步预测和实际物理反馈，在同一条真实仿真轨迹中逐步完成目标位姿。

## 1. 任务与算法

环境以 20 Hz 执行七维 OSC 动作（3 维位置增量、3 维旋转增量、1 维夹爪指令）。任务成功条件为 T 块与目标轮廓 IoU ≥ 0.85，物体直立且在桌面上；采集器另外要求连续 10 步成功，且末端高度 ≥ 0.895 m。默认 1000 步是防止死循环的保护上限，不是任务的时间要求。

1. **分阶段控制。** 先根据角度误差选择直接平移、连续旋转或大角度圆弧旋转，然后由标定接触规划器尝试平移和精细修正（此阶段保留原有控制约束）。成功则结束；若无法继续，在确认末端脱离接触后交给闭环控制器。
2. **闭环修正。** 在目标坐标系计算当前误差 q = (Δx, Δy, Δθ)。较大剩余角度优先尝试旋转；否则利用 Joint Lookahead 对候选接触点及推动距离进行最多四次接触的数学预测。考虑可达范围、时间估计、预测 IoU 和重复访问历史，只执行选中的第一个动作。
3. **实际物理反馈。** 每次推动包含接近、下降、推动、撤离与抬升。根据实际位移与转角，更新当前 episode 的局部接触响应，避免重复尝试停滞动作；接近目标时根据实测运动调整停止距离。随后从真实当前位置继续规划，不重置场景。
4. **成功筛选。** 满足稳定成功和末端抬升条件后保存原始连续轨迹；若连续六次接触无显著历史最佳改善、没有新接触、物体无效或保护预算耗尽，则退出且不保存为专家数据。

接触响应模型以物体局部坐标系中的单位正向物体行程产生的平移和旋转为参数。模型是近似预测器，不是训练好的神经网络；局部学习仅作用于当前任务，不跨任务更新标定文件。

## 2. 环境准备

建议使用已有能够运行 Push-T 的 Python 环境。`requirements.txt` 列出直接依赖，而非严格版本锁。

```bash
python check_setup.py
```

## 3. 数据采集

`collect_expert.py` 唯一正式策略为前向闭环。默认运行 24 个 0–90° 场景，最多 1000 步。

```bash
# 24 个大角度场景（90–180°），与已测试种子一致
python collect_expert.py --fresh --wide-angles --per-bin 4 --workers 4 --no-video

# 600 个 0–90° 场景
python collect_expert.py --collect --per-bin 100 --workers 4 --no-video

# 600 个 90–180° 场景
python collect_expert.py --collect --wide-angles --per-bin 100 --workers 4 --no-video

# 24 个大角度场景，仅录制其中 4 个代表性案例
python collect_expert.py --fresh --wide-angles --per-bin 4 --video --video-count 4
```

每个 15° 区间均衡采样正负角度。`--per-bin` 必须是正偶数；`--seed` 指定起始种子；`--max-steps` 可以调整保护上限；`--cases-file` 可复用已有 `cases.csv` 中的 seed 和 angle_deg；`--output` 指定新结果目录；`--plan-only` 仅生成场景清单。

多进程采集建议从 2–4 个 worker 开始。录像使用单 worker，会对最终选中轨迹做一次额外重放并核对终点状态；录像不是搜索过程，也不参与专家决策。正式批量采集关闭录像可避免额外开销。`--fresh` 标记评估候选，`--collect` 标记训练候选；均仅保存稳定成功的轨迹。

## 4. 输出数据

结果目录含 `cases.csv`、`summary.csv`、`corrections.csv`（存在闭环修正时）、`run_config.json`、`case_reports/`、`traces/` 和成功案例的 `demos/`。录像开启时另有 `videos/`。每个成功案例保存一个压缩 NPZ：

| 字段 | 形状/含义 |
|---|---|
| `actions` | (T, 7)，真实执行的归一化控制动作 |
| `obs/<key>` | (T+1, ...)，原始数值观测 |
| `stage`, `phase` | (T,)，执行阶段标签（保留原数据标签兼容性） |
| `success`, `iou` | (T+1,)，环境成功及轮廓重叠率 |
| `task_target_xy_yaw` | (T+1, 3)，目标位置与朝向 |
| `metadata_json` | 种子、角度、步数、成功阈值、标定文件哈希等 |

训练对齐使用 `obs[t] -> actions[t]`，而 `obs[t+1]` 是执行结果。数据为变长状态轨迹，不是同步视觉训练集；MP4 仅用于诊断。请按完整 episode/seed 划分训练、验证和测试集，并检查不同初始角度区间的成功数据覆盖。采集尝试数并不等于成功轨迹数。

## 5. 项目结构

```text
push_t_expert/
├── collect_expert.py          # 唯一运行入口
├── check_setup.py             # 环境与标定检查
├── requirements.txt
├── README.md
├── config/
│   ├── contact_model.json
│   └── contact_response_model.json
└── src/
    ├── collection.py          # 批量采样、并行、结果管理
    ├── forward_expert.py      # 单轨迹闭环、局部经验、动作执行
    ├── baseline.py            # 初始分阶段控制（内部必要模块）
    ├── rotation_controller.py
    ├── translation_controller.py
    ├── contact_planner.py     # 标定响应及分阶段规划
    ├── joint_lookahead.py     # 多步数学预测
    ├── geometry.py            # T 形轮廓、接触、坐标系
    ├── environment.py         # Panda Push-T 环境
    └── dataset.py             # 轨迹记录、导出、录像
```

## 6. 实验参考

在 24 个 90–180° 大角度场景（seed 9200–9223，保护上限 1000 步）中，成功 23 个（95.8%）：11 个由分阶段控制完成，12 个由闭环修正完成。成功轨迹平均 477.6 步；全部案例累计计算耗时 148.27 s。
