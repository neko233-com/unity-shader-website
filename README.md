# 🧠 AI Learning Hub

[![Deploy to GitHub Pages](https://github.com/neko233-com/ai-learning-website/actions/workflows/deploy.yml/badge.svg)](https://github.com/neko233-com/ai-learning-website/actions/workflows/deploy.yml)

系统学习AI知识的交互式平台，包含基础知识、代码实战和部署指南。

## 🌐 在线访问

**[https://neko233-com.github.io/ai-learning-website/](https://neko233-com.github.io/ai-learning-website/)**

## ✨ 特点

- 📚 **分类清晰** - 基础知识 / 多模态AI / 部署优化 三大模块
- 📖 **内容分层** - 每章包含：专业术语 → 基础概念 → 进阶知识 → 实战代码
- 🎯 **实用性强** - 包含大量可运行的代码示例
- 📱 **响应式设计** - 支持桌面和移动端

## 📖 内容目录

### 基础知识
| 章节 | 内容 |
|------|------|
| 深度学习基础 | 神经网络、反向传播、优化器、正则化 |
| Transformer架构 | 注意力机制、多头注意力、位置编码 |
| 大语言模型 | 预训练、微调、RLHF、Prompt工程 |

### 多模态AI
| 章节 | 内容 |
|------|------|
| 视觉理解 | ViT、CLIP、对比学习、零样本分类 |
| 视觉语言模型 | LLaVA、GPT-4V、视觉指令微调 |
| 图像生成 | 扩散模型、Stable Diffusion、ControlNet |
| 语音与音频 | Whisper、TTS、语音克隆 |

### 部署与优化
| 章节 | 内容 |
|------|------|
| 模型优化 | 量化、剪枝、知识蒸馏、推理加速 |
| 模型部署 | vLLM、Docker、K8s、负载均衡 |

## 🏗️ 项目结构

```
ai-learning-website/
├── index.html                     # 主页面
├── src/
│   ├── css/main.css              # 样式文件
│   ├── js/app.js                 # 应用逻辑
│   └── data/knowledge-base.json  # 知识库数据
├── docs/
│   └── LOCAL_MODEL_GUIDE.md      # 本地模型指南
├── scripts/
│   ├── deploy.sh                 # Linux/Mac 部署脚本
│   └── deploy.bat                # Windows 部署脚本
└── .github/workflows/deploy.yml  # GitHub Actions
```

## 🚀 一键部署

### Windows
```batch
scripts\deploy.bat "feat: 添加新章节"
```

### Linux/Mac
```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh "feat: 添加新章节"
```

## 🛠️ 本地运行

```bash
# 方式1: 直接打开
open index.html

# 方式2: 本地服务器
python -m http.server 8080
# 访问 http://localhost:8080
```

## 📝 添加新内容

编辑 `src/data/knowledge-base.json`：

```json
{
  "id": "new-chapter",
  "title": "新章节",
  "icon": "🆕",
  "sections": {
    "terminology": {
      "title": "专业术语",
      "items": [
        {"term": "术语", "english": "Term", "desc": "描述"}
      ]
    },
    "basic": {
      "title": "基础概念",
      "content": "### Markdown 内容"
    },
    "advanced": {
      "title": "进阶知识",
      "content": "### 进阶内容"
    },
    "practice": {
      "title": "实战代码",
      "content": "### 代码示例"
    }
  }
}
```

## 📚 相关资源

- [本地特调模型实现指南](./docs/LOCAL_MODEL_GUIDE.md)

## 📄 License

MIT

---

Made with ❤️ by [neko233](https://github.com/neko233-com)
