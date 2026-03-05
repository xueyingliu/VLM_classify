
# ACL-fig ResNet50 推理脚本（infer.py）

对科学图片数据进行分类：读取图片 → ResNet50 前向推理 → 打印预测类别与概率。  

**backbone是 ResNet50**

---

## 1. 环境依赖

- Python 3.8+
- PyTorch
- torchvision
- Pillow

安装示例（按你的环境自行调整）：

```bash
pip install torch torchvision pillow
```

## 2. 使用方法

### 1) 单张图片推理

```bash
python infer.py --input /path/to/image.png --ckpt best_aclfig_backbone_small_0.9076.pth 
```