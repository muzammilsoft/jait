# JAIT — 在浏览器中运行 TAIT 模型

JAIT 是用 TypeScript 编写的引擎，可直接在浏览器中运行 TAIT 训练的
`.npz` 模型。无需服务器、无需账号、数据不会离开设备：模型直接从
Hugging Face 下载（或从用户自己的文件打开），并在本地执行。

English: [README.md](README.md) · العربية: [README.ar.md](README.ar.md)

一个 `.npz`，三种运行时：

- **TAIT** 训练（Python / NumPy，在手机上）——
  [muzammilsoft/tait](https://github.com/muzammilsoft/tait)
- **KAIT** 在手机上运行（Kotlin）——
  [muzammilsoft/KAIT-Studio](https://github.com/muzammilsoft/KAIT-Studio)
- **JAIT** 在浏览器中运行（TypeScript）——本仓库

## 性能

`forward()` 不分配任何内存：所有中间张量都保存在每个模型专用的、
可重复使用的暂存区中，因此一次回复不会产生任何垃圾回收压力
（修复前，一次前向传播会分配 12 个 `Float64Array`——每个简短回复
约 0.5 MB 垃圾；现在前向传播为 0）。返回的概率是该暂存区的一个
视图：在对同一模型下一次调用 `forward()` 之前，请先使用或复制它们。

## 已验证

第一阶段关卡——`npm test` 加载 TinyBili
（`muzammilsoft/tinybili`，`tinyGPT0.1.npz`，1,072,206 字节）并与
Python TAIT 2.3.0 对比：

- 阿拉伯语和英语提示的编码器 ID 完全一致
- 前 5 个下一个 token 的 ID 一致，最大概率差为 **2.22e-16**
  （关卡：小于 1e-3）
- 贪心解码与 Python 完全一致
- 非 TAIT 文件会被明确报错并拒绝

测试在没有本地副本时从 Hugging Face 加载
（`JAIT_TEST_NPZ=/path/to/model.npz` 可覆盖来源）。
参考值位于 `test/reference.json`，由 `workspace/tait-dev` 中的真实
TAIT 源码生成。

## 模型格式（JAIT 实现的内容）

逐行移植自 `tait/core/model.py` 和 `tait/core/tokenizer.py`：

- **NPZ**：`.npy` 数组的 ZIP（STORED 或 DEFLATE）。权重：
  `wte, wpe, Wq, Wk, Wv, Wo, W1, b1, W2, b2, Whead, bhead`，另加
  BPE 数据 `merge_a, merge_b, merge_id` 和标量 `sep_id, end_id,
  tok_pad_id, block_size, pad_id`。`wte`/`wpe` 为 float32，其余为
  float64；合并为 int32。
- **分词器**：字节级 BPE。词表为 256 个单字节 + 合并 + 3 个特殊
  token（TinyBili 中 SEP 1000、END 1001、PAD 1002）。编码时优先合并
  合并 ID 最小的词对。NPZ 文件是自包含的：永远不需要 `vocab.json`。
- **模型**：单个 Transformer 块，**无 LayerNorm**（TAIT 没有）。带
  `-1e9` 掩码的因果自注意力、ReLU MLP、残差连接。嵌入相加时用
  `Math.fround` 舍入，因为 NumPy 以 float32 相加 float32 嵌入。
- **生成**：与 Python 顺序相同——PAD 置零、温度幂、top-k、top-p，
  然后抽取一次。TinyBili 是小型实验模型（6,666 个样本、50 轮），
  因此其回答常常是乱码；JAIT 诚实地将其展示为实验模型。

## 与 Python 的诚实差异

Python 使用 NumPy 的 PCG64 RNG 采样；JavaScript 没有等价物。
JAIT 使用 mulberry32，因此同一随机种子在 JS 中得到与 Python
*不同但可重复* 的采样。确定性比较必须使用前 5 概率或贪心解码，
绝不能用采样种子。

## 用法

安装（JAIT 尚未发布到 npm——请从 GitHub 安装；安装时会自动构建）：

```sh
npm install github:muzammilsoft/jait
```

或自行克隆并构建：

```sh
git clone https://github.com/muzammilsoft/jait
cd jait
npm install
npm run build   # -> dist/
npm test        # build + Phase 1 gate
```

然后：

```ts
import { loadModelFromUrl, generateReply } from 'jait';

const { model, tokenizer } = await loadModelFromUrl(
  'https://huggingface.co/muzammilsoft/tinybili/resolve/main/tinyGPT0.1.npz',
);
const reply = generateReply(model, tokenizer, 'مرحبا', {
  temperature: 0.8,
  topK: 40,
  topP: 0.95,
  maxNewTokens: 64,
});
```

或从用户自己的文件加载：

```ts
import { loadModelFromNpz } from 'jait';
const bytes = new Uint8Array(await file.arrayBuffer());
const { model, tokenizer } = loadModelFromNpz(bytes);
```

运行时依赖：仅 `fflate`（MIT），用于 ZIP/DEFLATE。TypeScript 仅为
开发工具；输出为普通 JavaScript。

JAIT 是 Khartoum Ghoul 的独立开源项目，与 Bilibili、Hugging Face
或任何模型托管方无关。
