# 计算机图形学实验

这是计算机图形学课程的实验项目集合，包含多个实验作业。

## 实验列表

| 实验 | 主题 | 内容 |
|------|------|------|
| **Work3** | 贝塞尔曲线 | 实现贝塞尔曲线的生成与渲染 |
| **Work4** | Phong 光照模型 | 实现 Phong 光照模型，包括环境光、漫反射、镜面反射 |
| **Work5** | 光线追踪 | 实现简单的光线追踪渲染器 |
| **Work6** | 可微渲染 | 使用 PyTorch3D 实现基于剪影的 3D 模型优化 |
| **Work7** | 质点弹簧模型 | 实现布料模拟，包含剪切弹簧、弯曲弹簧和球体碰撞 |
| **Work8** | SMPL 模型与 LBS | 实现 SMPL 骨骼蒙皮动画系统 |

## 环境要求

- Python 3.8+
- Taichi 1.7+（Work5、Work7）
- PyTorch（Work6、Work8）
- PyTorch3D（Work6）
- smplx（Work8）

## 运行方式

每个实验都有独立的 README.md 文件，详细说明了运行方式和参数配置。

```bash
# 进入对应实验目录
cd Work3
python 贝塞尔曲线.py

cd Work4
python Phong光照模型.py

cd Work5
python 光线追踪.py

cd Work6
python 可微渲染.py

cd Work7
python 质点弹簧模型.py

cd Work8
python run_lbs_lab.py
```

## 项目结构

```
CG-lab/
├── Work3/           # 贝塞尔曲线
│   ├── README.md
│   ├── 贝塞尔曲线.py
│   └── 贝塞尔曲线.gif
├── Work4/           # Phong 光照模型
│   ├── README.md
│   ├── Phong光照模型.py
│   └── Phong光照模型.gif
├── Work5/           # 光线追踪
│   ├── README.md
│   ├── 光线追踪.py
│   └── 光线追踪.gif
├── Work6/           # 可微渲染
│   ├── README.md
│   ├── 可微渲染.py
│   ├── 可微渲染.gif
│   └── cow.obj
├── Work7/           # 质点弹簧模型
│   ├── README.md
│   ├── 质点弹簧模型.py
│   └── 质点弹簧系统.gif
├── Work8/           # SMPL 模型与 LBS
│   ├── README.md
│   └── run_lbs_lab.py
└── .gitignore
```

## 注意事项

1. **模型文件**：Work6 和 Work8 需要额外的模型文件（`cow.obj`、`SMPL_NEUTRAL.pkl`），这些文件较大，请自行下载。
2. **依赖安装**：部分实验需要额外安装依赖，详见各实验的 README.md。
3. **GPU 加速**：Work5、Work7 使用 Taichi 进行 GPU 加速，Work6、Work8 使用 PyTorch，建议使用 GPU 运行以获得更好的性能。

## 许可证

MIT License
