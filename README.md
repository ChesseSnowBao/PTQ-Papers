# PTQ-Papers

- 包括了题目，会议/时间，机构，主要方法/贡献，所用的模型。

---
## Overview

- [PTQ-Papers ](#PTQ-Papers-)
  - [Overview](#overview)
  - [Survey](#survey)
  - [Transformer-based Models](#transformer-based-models)
    - [Vision Transformers](#vision-transformers)
    - [Language Transformers](#language-transformers)
  - [Convolutional Neural Networks](#convolutional-neural-networks)
    - [Visual Generation](#visual-generation)
    - [Image Classification](#image-classification)
    - [Other Tasks](#other-tasks)
      - [Object Detection](#object-detection)
      - [Super Resolution](#super-resolution)
      - [Point Cloud](#point-cloud)
  - [References](#references)



---


## Survey
- "A Survey of Quantization Methods for Efficient Neural Network Inference", Book Chapter: Low-Power Computer Vision, 2021. [[paper](https://arxiv.org/abs/2103.13630)]
- "Full Stack Optimization of Transformer Inference: a Survey", arXiv, 2023. [[paper](https://arxiv.org/abs/2302.14017)]
- "A White Paper on Neural Network Quantization", arXiv, 2021. [[paper](https://arxiv.org/abs/2106.08295)]
- "Binary Neural Networks: A Survey", PR, 2020. [[Paper](https://arxiv.org/abs/2004.03333)] [**`Extreme`**]


## Transformer-based Models
### Vision Transformers
- #### LRP-QViT: Mixed-Precision Vision Transformer Quantization via Layer-wise Relevance Propagation. [[paper](http://arxiv.org/abs/2401.11243)]
	- 会议/时间：arXiv, 2023
	- 机构：Rochester Institute of Technology
	- 模型：ViT, DeiT, Swin
	- 方法：1.用层级相关性传播对所有层分配得分，以得分分配不同层量化位宽。<br>2.改进RepQ-ViT：引入剪裁的channel-wise量化，用于经过LayerNorm的激活，以消除离群值，并改善固定位宽和混合精度量化的性能。

-  #### MPTQ-ViT: Mixed-Precision Post-Training Quantization for Vision Transformer. [[paper](http://arxiv.org/abs/2401.14895)] 
	-  会议/时间：arXiv, 2023
	-  机构：National Taiwan University
	- 模型：ViT, DeiT, Swin
	-  方法：MPTQ-ViT解决了Transformers的PTQ混合精度量化的几个问题：<br>1.通过使用带有bias的平滑量化，解决了激活的不对称分布减少clamping loss的问题；<br>2.提出了一种基于数据的机制，用于自动确定post-GeLU值的缩放因子(SFs)；<br>3.设计了一种综合考虑模型性能和压缩性能的选择度量，使用贪婪策略逐层确定权重和激活的位宽。方法适用于ViT、DeiT和Swin等模型。

- #### RepQ-ViT: Scale Reparameterization for Post-Training Quantization of Vision Transformers. [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_RepQ-ViT_Scale_Reparameterization_for_Post-Training_Quantization_of_Vision_Transformers_ICCV_2023_paper.pdf)] [[code](https://github.com/zkkli/RepQ-ViT)]]
	- 会议/时间：ICCV, 2023
	- 机构：Institute of Automation
	- 模型：ViT, DeiT, Swin
	- 方法：将PTQ的量化和推断过程解耦。在训练时对post-LayerNorm和post-Softmax activations使用channel维的log√2量化来保持原始数据分布，通过可解释的尺度重参数化将其转换为简化的量化器应用于推理阶段，以适应推断时的硬件环境。

- #### NoisyQuant: Noisy Bias-Enhanced Post-Training Activation Quantization for Vision Transformers. [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_NoisyQuant_Noisy_Bias-Enhanced_Post-Training_Activation_Quantization_for_Vision_Transformers_CVPR_2023_paper.pdf)]
	- 会议/时间：CVPR, 2023
	- 机构：Nanjing University
	- 模型：ViT, DeiT, Swin, DETR
	- 方法：对于Transformers中激活分布造成PTQ性能下降的问题，发现和证明了在量化之前，通过向激活添加从均匀分布中采样得到的固定噪声偏差，可以显著减少量化误差。基于此开发了量化器无关的增强方法NoisyQuant，使用固定的加性噪声偏差减少重尾分布量化误差。

- #### Q-HyViT: Post-Training Quantization for Hybrid Vision Transformers with Bridge Block Reconstruction. [[paper](https://arxiv.org/abs/2303.12557))]
	- 会议/时间：arXiv, 2023
	- 机构：ETRI
	- 模型：MVv1, MVv2, MF, EFv1, EFv2, MobileViTv1, MobileViTv2
	- 方法：解决混合Transformers量化时的问题：高度动态的激活范围、桥接块中的零点溢出、多样化的归一方法、少于500万的参数数量。提出解决方法：基于Hessian矩阵来调整桥接层和非桥接层的粒度和方案，同时确定量化的最佳缩放因子。

- #### Patch Similarity Aware Data-Free Quantization for Vision Transformers.  [[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710154.pdf)] [[code](https://github.com/zkkli/psaq-vit)]
	- 会议/时间：ECCV, 2022
	- 机构：Institute of Automation
	- 模型：ViT-S, ViT-B, DeiT-T, DeiT-S, DeiT-B, Swin-T, Swin-S
	- 方法：解决PTQ在用于Transformers时生成校准样本的问题。通过分析自注意力模块在输入为真实图像时的patch相似性，利用其差分熵来量化响应的多样性，通过核密度估计计算，确保梯度反向传播。差分熵被用作目标函数，优化高斯噪声以逼近真实图像，以此生成样本数据。

- #### PTQ4ViT: Post-Training Quantization for Vision Transformers with Twin Uniform Quantization (2022). [[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720190.pdf)] [[code](https://github.com/hahnyuan/ptq4vit)] 
	- 会议/期刊：ECCV, 2022
	- 机构/大学：School of Computer Science, Peking University; School of Integrated Circuits, Peking University; Houmo AI
	- 模型：ViT-S, ViT-B, DeiT-S, DeiT-B, Swin-T, Swin-S, Swin-B
	- 背景：先前的训练后量化方法在视觉转换器上表现不佳，导致即使在8位量化中也有超过1%的精度下降。
	- 贡献：
		1. PTQ在视觉转换器上存在的问题是post-softmax和post-GELU激活的特殊分布和不准确的度量。
		2. 提出了Twin Uniform Quantization来处理这些特殊分布，它可以在现有的硬件设备（包括CPU和GPU）上有效地处理。
		3. 提出了使用Hessian引导的度量来确定最佳的缩放因子，这取代了不准确的度量。

- #### FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer.  [[paper](https://arxiv.org/abs/2111.13824)]  [[code](https://github.com/megvii-research/FQ-ViT)] 
	- 会议/期刊：IJCAI, 2022
	- 机构：MEGVII Technology
	- 模型：ViT, DeiT, Swin
	- 贡献：
	1. 提出Power-of-Two Factor，通过逐层量化实现对LayerNorm输入的准确量化。
	2. 提出了Log-Int-Softmax，对注意力图进行4位量化。将注意力图存储在极低的位数上，并用BitShift运算符替代乘法。在Softmax模块上实现了纯整数推断，极大减少了推理的资源消耗。

- #### Post-training quantization for vision transformer. [[paper](https://openreview.net/forum?id=9TX5OsKJvm)] 
  - 会议/期刊：NeurIPS, 2021
  - 机构/大学：1. Peking University 2. Noah's Ark Lab, Huawei Technologies 3. Peng Cheng Laboratory
  - 模型：ViT-B, ViT-L, DeiT-S, DeiT-B, DETR
  - 贡献：
    1. 引入ranking loss用以保持自注意力值的相对顺序。
    2. 根据特征多样性（通过注意力图和输出特征计算的核范数）确定每层的位宽。
  - 方法：交替搜索权重和输入的量化区间，并引入偏差校正（bias correction）。

- #### Towards accurate post-training quantization for vision transformer (2022 APQ-ViT)
  - 会议/期刊：ACMMM, 2022
  - 机构/大学：1. Beihang University 2. Meituan
  - 模型：ViT-T, ViT-S, ViT-S/32, ViT-B, DeiT-T, DeiT-S, DeiT-B, Swin-S, Swin-B, Swin-B/384
  - 背景：现有的训练后量化方法在ViT上仍然会造成严重的性能下降。
  - 贡献：
    1. 分析性能下降的原因：（1）现有的校准度量对极低比特表示的量化影响度量不准确；（2）量化范式与Softmax的幂律分布存在不一致性。
    2. 提出APQ-ViT，并采用了一种统一的分块自底向上的校准方案，以实现块内的量化感知，并对影响最终预测的关键误差进行优先排序。
    3. 揭示了Softmax函数的幂律分布，并提出了保持Matthew效应的量化方法。

- #### Tsptq-vit: Two-scaled post-training quantization for vision transformer (2023)
  - 会议/期刊：ICASSP, 2023
  - 机构/大学：National Taiwan University
  - 模型：ViT-S, ViT-B, ViT-L, DeiT-T, DeiT-S, DeiT-B, Swin-T, Swin-S, Swin-B
  - 背景：
    1. Softmax和GeLU后的值非正态分布（导致量化后的精度下降）
    2. LayerNorm中存在高通道方差
  - 贡献：提出了一个两尺度的训练后量化方案：
    1. Value-Aware Two-Scaled Scaling Factors (V-2SF)：针对post-Softmax和post-GeLU值的非正态分布，提出了一种双尺度策略，利用比特稀疏性进行量化。
    2. Outlier-Aware Two-Scaled Scaling Factors (O-2SF)：设计了另一种双尺度机制，用于处理LayerNorm的输入通道维度，以缓解离群值的主导影响。


- #### Mr.biq: Post-training nonuniform quantization based on minimizing the reconstruction error (2022) [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Mr.biq/Mr.biq_Paper.pdf)]
	- 会议/期刊：CVPR, 2022
	- 大学/机构：1. Samsung Research, 2. Korea University, 3. The University of Texas at Austin
	- 模型：ViT-B, ViT-L, DeiT-S, DeiT-B, BERT, DistilBERT
	- 贡献：
		- 提出了一种新的训练后非均匀量化方法，称为Mr. BiQ，即使在Transformer模型上也允许低位宽量化。
		- 对权重进行多级二值化，同时允许激活被表示为各种数据格式。与传统方法不同，先优化全精度权值然后将权值分解为量化参数，Mr. BiQ将量化参数（即比例因子和区位码）识别为优化过程中可直接联合学习的参数。


[[Back to Overview](#overview)]

### Language Transformers
- #### APTQ: Attention-aware Post-Training Mixed-Precision Quantization for Large Language Models, arXiv, 2024. [[paper](http://arxiv.org/abs/2402.14866)]

-  #### CBQ: Cross-Block Quantization for Large Language Models, arXiv, 2023. [[paper](http://arxiv.org/abs/2312.07950)]

- #### FP8-BERT: Post-Training Quantization for Transformer. [[paper](http://arxiv.org/abs/2312.05725)]
	- 会议/期刊：arXiv, 2023
	- 机构：North Carolina State University
	- 模型：BERT, ResNet, LLM Transformer
	- 贡献：
		- 在各种任务、模型和数据集上验证了FP8 PTQ方法的性能。
		- 提出了一种可靠的PTQ策略，适用于BERT和其他基于Transformer的模型。该策略与最广泛接受的PTQ INT8量化一样简单，同时能够更接近完全精度模型的准确性。

- #### QuIP: 2-Bit Quantization of Large Language Models With Guarantees. [[paper](https://neurips.cc/virtual/2023/poster/69982)] [[code](https://github.com/jerry-chee/QuIP)]
	- 会议/期刊：NeurIPS, 2023
	- 机构：Cornell University
	- 模型：OPT, LLaMA, LLM Transformer
	- 贡献：
		- 文章观察到权重和Hessian矩阵的不相干性有利于量化，提出了一种不相干处理的量化方法（QuIP），包括两个步骤：
			1. 最小化一个二次代理目标的自适应舍入过程。
			2. 通过随机正交矩阵的乘法实现高效的预处理和后处理，确保权重和Hessian矩阵的不相干性。
		- 该方法应用于几种现有的量化方法，使得在权重上进行2位量化成为可能。



- #### SqueezeLLM: Dense-and-Sparse Quantization. [[paper](https://arxiv.org/abs/2306.07629)]
	- 会议/期刊：arXiv, 2024
	- 机构：UC Berkeley
	- 模型：OPT, LLaMA, Vicuna, LLM Transformer
	- 贡献：
		- 证明了对于单批次推理，生成式LLM的瓶颈是内存带宽而非计算资源。
		- 提出了针对内存优化的3位PTQ框架，包括以下两个思想：
			1. 针对权重的非均匀分布，采用基于灵敏度的非均匀量化方法。
			2. 将权重分解为稠密和稀疏两个部分。稀疏部分使用高效的稀疏存储方法以完整精度保存离群值，而稠密部分则使用更低的位宽进行量化。


- #### QLLM: Accurate and Efficient Low-bitwidth Quantization for Large Language Models. [[paper](http://arxiv.org/abs/2310.08041)]
	- 会议/期刊：arXiv, 2024
	- 机构：Monash University
	- 模型：LLaMA, LLM Transformer
	- 贡献：
		- 针对激活的离群值，提出无梯度channel重组技术，将异常值channel分解为几个子channel。通过分散异常值的幅度，确保channel之间的激活范围均匀。然后将channel组装，将相似的channel融合在一起以保持原始channel数。
		- 鉴于不同层之间离群值模式的差异以及极端离群值的存在，提出基于最小化原始输出激活和重新组装输入激活之间的重组误差的策略，确定每层的最佳拆解channel数。


- #### FPTQ: Fine-grained Post-training Quantization for Large Language Models. [[paper](http://arxiv.org/abs/2308.15987)]
	- 会议/期刊：arXiv, 2023
	- 机构：Meituan
	- 模型：BLOOM, LLaMA, LLM Transformer
	- 贡献：
		- 结合W8A8和W4A16，提出了W4A8 PTQ方案，充分利用4bit权重量化的I/O利用率优势和8bit矩阵计算的加速优势。
		- 采用了逐层激活量化策略来应对性能下降的挑战，并针对最困难的层采用对数均衡方法，将其与细粒度权重量化相结合。

- #### FineQuant: Unlocking Efficiency with Fine-Grained Weight-Only Quantization for LLMs. [[paper](http://arxiv.org/abs/2308.09723)]
	- 会议/期刊：arXiv, 2023
	- 机构：Microsoft
	- 模型：GPT, OPT, LLM Transformer
	- 贡献：
		- 提出了一种高效的仅权重量化方法，包括细粒度量化算法，其中包括分组量化和自适应粒度选择。
		- 实现了高效的GPU GEMMs，能够进行即时的矩阵乘法和反量化操作，支持fp16或bf16激活与int8或int4权重的乘法。

- #### Gradient-Based Post-Training Quantization: Challenging the Status Quo, arXiv, 2023. [[paper](http://arxiv.org/abs/2308.07662)]

- #### OWQ: Outlier-Aware Weight Quantization for Efficient Fine-Tuning and Inference of Large Language Models. [[paper](http://arxiv.org/abs/2306.02272)]
	- 会议/期刊：arXiv, 2024
	- 机构：POSTECH
	- 模型：OPT, LLaMA, LLM Transformer
	- 贡献：
		- 提出了异常感知权重量化(OWQ)方法，将量化敏感的结构化权重存储为高精度，对其他权重应用高度优化的量化方法。使得3.1位模型性能与使用GPTQ的4位模型相当。
		- 使用参数高效的用于任务特定适应的微调方法，称为弱列微调（WCT），减小了内存开销。

- #### SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression. [[paper](https://arxiv.org/abs/2306.03078)]
	- 会议/期刊：arXiv, 2023
	- 机构：University of Washington
	- 模型：LLaMA, Falcon, LLM Transformer
	- 贡献：
		- 文章将离群权重使用高精度存储，而其他权重以低比特存储。
		- 实现了一种尺寸很小(16个连续元素)的分组量化变体，并将量化尺度本身量化为3比特的形式。
		- 介绍了将预训练LLM转换为SpQR格式的方法，展示了基于SpQR的稀疏矩阵乘法算法，并将其与3-4位权重的稠密量化矩阵乘法相结合的逐token生成方法。

- #### AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. [[paper](https://arxiv.org/abs/2306.00978)]
	- 会议/期刊：arXiv, 2023
	- 机构：MIT
	- 模型：OPT, LLaMA, LLM Transformer
	- 贡献：
		- 文章发现保证大约1%的主要权重的精度就可大大减少量化误差。
		- 基于此，通过激活的分布来找到主要权重的通道。
		- 设计了逐通道缩放的方法，搜索最佳缩放，使得在完全权重量化下的量化误差最小，同时避免过度拟合校准集。


- #### RPTQ: Reorder-based Post-training Quantization for Large Language Models, arXiv, 2023. [[paper](https://arxiv.org/abs/2304.01089)] [[code](https://github.com/hahnyuan/rptq4llm)]


- #### SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. [[paper](https://arxiv.org/abs/2211.10438)] [[code](https://github.com/mit-han-lab/smoothquant)]
	- 会议/期刊：ICML, 2023
	- 机构：MIT
	- 模型：OPT, BLOOM, GLM-130B, LLM Transformer
	- 贡献：
		- 由于存在离群值，激活比权重更难量化。SmoothQuant通过离线地将量化困难从激活迁移到权重上。
		- 提出了一个数学等价的逐通道缩放变换，显著平滑了通道间的幅度差异，使模型更适合量化。

- #### SmoothQuant+: Accurate and Efficient 4-bit Post-Training Weight Quantization for LLM. [[paper](http://arxiv.org/abs/2312.03788)]
	- 会议/期刊：arXiv, 2023
	- 机构：MIT
	- 模型：Llama, LLM Transformer
	- 贡献：
		- 在SmoothQuant的基础上，将权重量化到4位。

- #### GPTQ: Accurate Post-training Quantization for Generative Pre-trained Transformers. [[papar](https://arxiv.org/abs/2210.17323)]  [[code](https://github.com/IST-DASLab/gptq)]
	- 会议/期刊：ICLR, 2023
	- 机构：IST Austria
	- 模型：BLOOM, OPT, LLM Transformer
	- 贡献：
		- 设计了生成预训练Transformer模型的PTQ方法，对某个 block 内的所有参数逐个量化，并通过适当调整未量化参数来弥补量化引起的精度损失。该方法高效且精确，能够在最多几个小时内执行具有数千亿参数的模型，且位宽可以小到3-4位几乎不损失精度。
		- 证明了在极端情况下可以将每个组件(component)量化到2位或三值。
		- 开发了可以高效地在生成任务中执行压缩后的模型的执行框架。


- #### Outlier Suppression: Pushing the Limit of Low-bit Transformer Language Models. [[paper]](https://arxiv.org/abs/2209.13325) [[code](https://github.com/wimh966/outlier_suppression)]
	- 会议/期刊：NeurIPS, 2022
	- 机构：Beihang University
	- 模型：BERT, RoBERTa, BART, LLM Transformer
	- 贡献：
		- 因为LLM Transformer结构化离群值对量化精度影响极大，提出了以下方法：
			1. Gamma Migration：将LN层的γ（会放大输出的离群值）迁移到后续模块中。
			2. Token-Wise Clipping：粗粒度地跳过不重要的离群值，通过逐标记获得初步的剪裁范围，然后细粒度地对其进行优化。

- #### Outlier Suppression+: Accurate quantization of large language models by equivalent and effective shifting and scaling. [[paper](https://arxiv.org/abs/2304.09145)] 
	- 会议/期刊：arXiv, 2023
	- 机构：Beihang University
	- 模型：LLaMA, BERT, LLM Transformer
	- 贡献：
		- 针对离群值集中在特定通道中，并且在通道间存在不对称性，提出以下方法：
			1. 提出通道级偏移操作，并采用通道级缩放处理离群值的集中性属性。设计统一的迁移模式，将离群值的相反效果迁移到后续模块中。
			2. 确定有效的偏移和缩放值的稳定方法。偏移值消除通道之间的不对称特征，缩放值定量地最小化由迁移和量化引起的激活和权重之间的交互输出变化，通过快速稳定的搜索过程实现平衡的量化。

- #### Towards Efficient Post-training Quantization of Pre-trained Language Models, NeurIPS, 2022. [[paper](https://nips.cc/Conferences/2022/Schedule?showEvent=53407)]

- #### ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers, NeurIPS, 2022. [[paper](https://nips.cc/Conferences/2022/Schedule?showEvent=54407)] [[code](https://github.com/microsoft/DeepSpeed)]


[[Back to Overview](#overview)]

## Convolutional Neural Networks
### Visual Generation
- #### Efficient Quantization Strategies for Latent Diffusion Models, arXiv, 2023. [[paper](http://arxiv.org/abs/2312.05431)]
- "PTQD: Accurate Post-Training Quantization for Diffusion Models", NeurIPS, 2023. [[paper](https://neurips.cc/virtual/2023/poster/71314)] [**`PTQ`**]
- "Q-diffusion: Quantizing Diffusion Models", ICCV, 2023. [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Q-Diffusion_Quantizing_Diffusion_Models_ICCV_2023_paper.pdf)] [[code](https://github.com/Xiuyu-Li/q-diffusion)] [**`PTQ`**]
- #### Towards Accurate Data-free Quantization for Diffusion Models, arXiv, 2023. [[paper](http://arxiv.org/abs/2305.18723)]


- #### Post-training Quantization on Diffusion Models (PTQDM) [[paper](http://openaccess.thecvf.com/content/CVPR2023/html/Shang_Post-Training_Quantization_on_Diffusion_Models_CVPR_2023_paper.html)] [[code](https://https//github.com/42Shawn/PTQ4DM)]
	- 会议/期刊：CVPR, 2023
	- 大学/机构：1. Illinois Institute of Technology, 2. Houmo AI, 3. Tencent AI Lab, 4. Cisco Research
	- 任务：
		1. ImageNet 64x64 DDIM 100/250/4000 steps
		2. CIFAR 32x32 DDIM 100/250/4000 steps
	- 背景：最近，基于得分的去噪扩散生成模型在生成逼真和多样化数据方面取得了显著成就。然而，目前的去噪扩散模型生成过程缓慢，依赖于复杂的神经网络和冗长的迭代噪声估计，这限制了它在边缘设备上的广泛部署。
	- 贡献：
		1. 为了加速去噪扩散模型，引入了后训练量化（PTQ）到DM加速中，将噪声估计网络以训练后的方式直接量化。据我们所知，这是第一个从无训练网络压缩的角度研究扩散模型加速的工作。
		2. 通过对PTQ和DMs进行全面研究，发现PTQ导致DMs性能下降的原因是不同时间步输出分布的差异。针对这一观察，提出了PTQ4DM方法。
		3. 在实验中，首次将预训练的扩散模型量化为8位，且性能损失不明显。重要的是，PTQ4DM可以作为其他SoTA DM加速方法的即插即用模块。






### Image Classification

- #### Unified Data-Free Compression: Pruning and Quantization without Fine-Tuning.  [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Bai_Unified_Data-Free_Compression_Pruning_and_Quantization_without_Fine-Tuning_ICCV_2023_paper.pdf)]
	- 会议/期刊：ICCV, 2023
	- 机构：Zhejiang University
	- 模型：ResNet, MobileNet, DenseNet, VGG-16
	- 应用领域：CNN-Image Classification
	- 贡献：
		- 证明并实现了通过重构第(l + 1)层的channel来恢复第l层由于剪枝或量化channel引起的信息损失。
		- 表示了原始网络和其压缩网络之间的重构误差，并证明了该误差可以被最小化。


- #### FlexRound: Learnable Rounding based on Element-wise Division for Post-Training Quantization.[[paper](https://openreview.net/forum?id=EPnzNJTYsb)]
	- 会议/期刊：PMLR, 2023
	- 机构：NAVER Cloud
	- 模型：LLaMA, ResNet, MobileNetV2, BERT, GPT-Neo, OPT, GPT-2
	- 应用领域：CNN-Image Classification
	- 贡献：
		- 基于元素除法的可学习舍入方法，用于权重的后训练量化。
		- 可以学习一层或一个channel中共享的量化网格和每个权重的scale。


- #### Bit-shrinking: Limiting Instantaneous Sharpness for Improving Post-training Quantization. [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_Bit-Shrinking_Limiting_Instantaneous_Sharpness_for_Improving_Post-Training_Quantization_CVPR_2023_paper.pdf)]
	- 会议/期刊：CVPR, 2023
	- 机构：hikvision
	- 模型：Faster RCNN, RetinaNet, InceptionV3, MobileNetV2
	- 应用领域：CNN-Image Classification
	- 贡献：
		- 针对低位宽量化中损失曲面容易陷入较差的局部最小值的问题。
		- 定义锐度项（sharpness term）来估计量化噪声对损失的影响。
		- 设计了Bit-shrinking方法，将位宽设定为连续值，通过保持锐度项的小和稳定，找到更好的局部最小值。


- #### Solving Oscillation Problem in Post-Training Quantization Through a Theoretical Perspective. [[paper](https://arxiv.org/pdf/2303.11906.pdf)] [[code](https://github.com/bytedance/mrecg)]
	- 会议/期刊：CVPR, 2023
	- 机构：Xiamen University
	- 模型：ResNet, MobileNet
	- 应用领域：CNN-Image Classification
	- 贡献：
		- 证明了PTQ（Post-Training Quantization）重建过程中损失的振荡是由模块容量的差异引起的。
		- 提出混合重构粒度（MRECG）方法，利用损失度量和模块容量来优化数据依赖和无数据场景下的混合重构细粒度，解决了PTQ中的振荡问题。


- #### GENIE: Show Me the Data for Quantization. [[paper](https://arxiv.org/abs/2212.04780)] [[code](https://github.com/SamsungLabs/Genie)]
	- 会议/期刊：CVPR, 2023
	- 机构：Samsung Research
	- 模型：ResNet, MobileNet, RegNet, MnasNet
	- 应用领域：CNN-Image Classification
	- 贡献：
		- 将PTQ（Post-Training Quantization）用于Zero-shot量化，无需真实数据集，仅需几小时即可完成量化过程。
		- 提出GENIE框架，通过知识蒸馏生成适用于量化的数据。


- #### Mr.BiQ: Post-Training Non-Uniform Quantization based on Minimizing the Reconstruction Error.  [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Jeon_Mr.BiQ_Post-Training_Non-Uniform_Quantization_Based_on_Minimizing_the_Reconstruction_Error_CVPR_2022_paper.pdf)]
	- 会议/期刊：CVPR, 2022
	- 机构：Samsung Research
	- 模型：BERT, DistilBERT
	- 应用领域：CNN-Image Classification
	- 贡献：
		- 提出了基于重建误差最小化的PTQ（Post-Training Quantization）的多级二进制非均匀量化方法。
		- 学习优化量化参数以获得量化权重。
		- 对量化模型进行全面搜索，在极低位宽（如W2A42或W2A8）下，准确性几乎没有下降。


- #### Leveraging Inter-Layer Dependency for Post-Training Quantization.  [[paper](https://nips.cc/Conferences/2022/Schedule?showEvent=54389)]
	- 会议/期刊：NeurIPS, 2022
	- 机构：Ant Technology Group
	- 模型：ResNet, MobileNet, RegNet, MnasNet
	- 应用领域：CNN-Image Classification
	- 贡献：
		- 解决了利用层间依赖的两个问题：
			1. 利用激活正则化（Activation Regularization，AR）缓解过拟合。
			2. 引入模拟退火Softmax（Annealing Softmax，ASoftmax）和模拟退火Mixup（Annealing Mixup，AMixup）优化离散变量。


- #### Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning.[[paper](https://nips.cc/Conferences/2022/Schedule?showEvent=53412)] [[code](https://github.com/ist-daslab/obc)]
	- 会议/期刊：NeurIPS, 2022
	- 机构：IST Austria
	- 模型：ResNet, YOLO v5, BERT
	- 应用领域：CNN-Image Classification
	- 贡献：
		- 开发了逐层量化和剪枝的框架OBC（Optimal Brain Compression）。
		- 使用OBS（Optimal Brain Surgeon）计算原始层输出和压缩后的输出之间的平方差||Wx - W'x'||^2，计算复杂度降低到O(d · dcol^2)，其中d为层维度，dcol为权重矩阵列维度。
		- 可以在合理的时间内在单个GPU上对千万级参数实现精确贪婪解。
		- 将OBS用于量化，迭代逐个量化权重，并对剩余未量化权重进行更新。


- #### Non-Uniform Step Size Quantization for Accurate Post-Training Quantization. [[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710657.pdf)]
	- 会议/期刊：ECCV, 2022
	- 机构：UNIST
	- 模型：ResNet, InceptionV3, MobileNetV2
	- 应用领域：CNN-Image Classification
	- 贡献：
		- 针对低位量化中PTQ（Post-Training Quantization）方法相对于QAT（Quantization Aware Training）方法的性能差异，在两字对数尺度量化背景下提出了子集量化（SQ）的概念。
		- 提出了一种方法，可以将量化器指定为量化点的子集，而不是一个算术函数。
		- 在给定数据分布下，提出了一种方法来找到最佳的子集和缩放因子。

- #### RAPQ: Rescuing Accuracy for Power-of-Two Low-bit Post-training Quantization. [[paper](https://www.ijcai.org/proceedings/2022/219)] [[code](https://github.com/billamihom/rapq)]
	- 会议/期刊：IJCAI, 2022
	- 机构：Peking University
	- 模型：ResNet, MobileNet, RegNet
	- 应用领域：CNN-Image Classification
	- 贡献：
		- 提出了低位二次幂PTQ（Post-training Quantization）框架RAPQ，旨在解决二次幂的缩放因子可选值较少的问题。
		- 动态调整整个网络的缩放因子，而不是静态地逐层确定，从而提高量化精度。

- #### Quantization and training of neural networks for efficient integer-arithmetic-only inference.
	- 会议/期刊：CVPR, 2018
	- 机构/大学：Google Inc.
	- 模型：ResNet-50, ResNet-100, ResNet-150
	- 任务：ImageNet classification and COCO detection
	- 贡献：
		- 提供了一个可在如Qualcomm Hexagon等纯整数算术硬件上高效实现的量化推理框架，并描述了一个在ARM NEON上高效、准确的实现。
		- 提供了一个与量化推理共同设计的量化训练框架，以最小化量化对真实模型的精度损失。



- #### QDrop: Randomly Dropping Quantization for Extremely Low-bit Post-Training Quantization [[paper](https://openreview.net/forum?id=ySQH0oDyp7)] [[code](https://github.com/wimh966/QDrop)]
	- 会议/期刊：ICLR, 2022
	- 大学/机构：1. Beihang University, 2. SenseTime Research
	- 模型：Faster RCNN, RetinaNet, Res18, Res50, MNV2, Reg600M, Reg3.2G, MNasx2
	- 贡献：
		- 证实了在PTQ（Post-Training Quantization）重建中引入激活量化的好处，并观察到激活量化的部分参与比整体参与表现更好。
		- 建立了一个理论框架来深入分析将激活量化纳入权重调整的影响。使用该框架，得出结论，优化后的低比特模型在校准数据和测试数据上的平坦度对最终的精度至关重要。
		- 提出了一个简单而有效的方法QDROP，该方法从一般的角度实现了平坦性。QDROP易于实现，并且可以持续地将现有方法作为各种神经网络的即插即用模块，包括像深度残差网络这样的CNN和像BERT这样的Transformers。


- #### Accurate post-training quantization with small calibration sets (AdaQuant) [[paper](http://proceedings.mlr.press/v139/hubara21a.html)] [[code](https://github.com/papers-submission/CalibTIP)]
	- 会议/期刊：ICML, 2021
	- 大学/机构：1. Habana Labs – An Intel company, 2. Department of Electrical Engineering - Technion
	- 模型：RN-18, RN-34, RN-50, RN-101, RNext-50, Inc-V3, MobileNet, BERT-Base
	- 背景：当使用低于8位的量化（除了在小数据集上）时，目前的量化方法往往会导致显著的精度下降。
	- 贡献：
		- 提出了AdaQuant，一种逐层优化方法，旨在最小化量化层输出与全精度层输出之间的误差。该方法仅需要少量的标定数据集，无需过拟合即可从训练数据中获得。
		- 引入整数规划，因为网络的某些部分可能允许比其他层更低的精度。建议使用基于整数线性规划的方法来确定不同层的精度水平。该方法的目标是在不违反预定义的网络精度下降或压缩约束的前提下，最大化期望的加速比或功耗节约。
		- 提出了Para-normalization，在量化之后观察到批量范数统计量的均值和方差存在固有的偏差。通过在批量归一化中使用重新估计的统计量，可以恢复大部分量化网络的性能。
		- 分析了每种方法的优缺点，并提出了两种流水线：（1）轻流水线，不需要反向通道，因此可以在仅有推理硬件的情况下调用；（2）高级流水线，包括AdaQuant和偏置调优。


- #### Post-Training Sparsity-Aware Quantization, NeurIPS, 2021. [[paper](https://openreview.net/forum?id=qe9z54E_cqE)] [[code](https://github.com/gilshm/sparq)]

- #### Diversifying Sample Generation for Accurate Data-Free Quantization, CVPR, 2021. [[paper](https://arxiv.org/abs/2103.01049)]

- #### BRECQ: pushing the limit of post-training quantization by block reconstruction. [[paper](https://openreview.net/forum?id=POWv6hDd9XH)] [[code](https://github.com/yhhhli/BRECQ)]
	- 会议/期刊：ICLR, 2021
	- 大学/机构：1. University of Electronic Science and Technology of China, 2. SenseTime Research
	- 模型：ResNet-18, ResNet-50, MobileNetV2, RegNet-600MF, RegNet-3.2GF, MnasNet-2.0
	- 贡献：
		- 在二阶分析的基础上，定义了一组重构单元，并在理论和经验证据的支持下表明，分块重构是最佳选择。使用Fisher信息矩阵为每个预激活分配一个重构过程中的重要性度量。
		- 结合遗传算法和定义良好的块内敏感度度量，生成延迟和尺寸保证的混合精度量化神经网络。这种方法实现了在专用硬件（FPGA）和通用硬件（ARM CPU）上的普遍改进。
		- 发现该方法适用于各种任务和模型。此外，首次证明了训练后量化可以将权重量化到INT2中，而无需显著的精度损失。


- #### Post-training Quantization with Multiple Points: Mixed Precision without Mixed Precision, AAAI, 2021. [[paper](https://arxiv.org/pdf/2002.09049)]




- #### ZeroQ: A Novel Zero Shot Quantization Framework [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shen_A_Novel_Zero-Shot_Quantization_Framework_CVPR_2020_paper.pdf)] [[code](https://github.com/PKU-DAIR/ZeroQ)]
	- 会议/期刊：CVPR, 2020
	- 大学/机构：1. Peking University, 2. University of California, Berkeley
	- 模型：ResNet50, MobileNetV2, ShuffleNet
	- 贡献：
		- 提出了一个优化公式来生成蒸馏数据，即设计合成数据来匹配批归一化层的统计信息。这种重构具有较小的计算开销。
		- 使用上述重构框架对量化后的模型和原始模型进行敏感性分析，证明了蒸馏后的数据与原始训练数据的敏感度相匹配。然后，使用蒸馏数据代替原始/真实数据进行训练后的量化。这一过程中的整个敏感度计算对ResNet50网络只需要12秒（仅占一个epoch训练时间的0.2%）。重要的是，在整个过程中，没有使用任何训练/验证数据。
		- 本文的框架同时支持均匀量化和混合精度量化。对于后者，提出了一种基于Pareto前沿优化的自动精度选择方法。该方法通过计算基于蒸馏数据的量化敏感度实现，并具有较小的计算开销。


- #### Data-free quantization through weight equalization and bias correction (DFQ) [[paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Nagel_Data-Free_Quantization_Through_Weight_Equalization_and_Bias_Correction_ICCV_2019_paper.html)] 
	- 会议/期刊：ICCV, 2019
	- 大学/机构：1. Qualcomm AI Research, 2. Qualcomm Technologies Netherlands B.V.
	- 模型：MobileNetV2, MobileNetV1, ResNet18
	- 贡献：
		- 引入了一种不需要数据、微调或超参数调优的量化方法，通过简单的 API 调用实现了精度提升。
		- 该方法利用激活函数的尺度等变性特性来均衡网络中的权重范围。此外，纠正了量化过程中引入的误差偏差。
		- 进一步表明，该量化方法还适用于其他计算机视觉架构和任务，如语义分割和目标检测。


- #### Post-training 4-bit quantization of convolutional networks for rapid deployment.
	- 会议/期刊：NeurIPS, 2019
	- 大学/机构：1. Intel – Artificial Intelligence Products Group (AIPG), 2. Technion – Israel Institute of Technology
	- 模型：VGG, VGG-BN, IncepV3, Res18, Res50, Res101
	- 背景：卷积神经网络的量化可以减少中间结果的存储和计算量，但通常需要完整的数据集和耗时的微调来恢复由量化引起的精度损失。
	- 贡献：
		- 提出了一种名为 Analytical Clipping for Integer Quantization (ACIQ) 的方法，通过限制张量内的激活值范围来减少舍入误差。ACIQ 方法通过解析地近似最优裁剪值，最小化均方误差度量。这种分析阈值在运行时使用简单，并且可以轻松与其他量化技术集成。
		- 引入了一种逐通道比特分配策略，以确定每个通道的最佳比特宽度。通过分析解决了这个问题，并证明了在对输入分布做出一些假设的情况下，每个通道的最佳量化步长与其范围的 2/3 次方成正比。
		- 观察到量化后的权重值存在固有偏差，提出了一种简单的偏差校正方法来补偿这种偏差。


- #### Quantization and training of neural networks for efficient integer-arithmetic-only inference [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf)]
	- 会议/期刊：CVPR, 2018
	- 机构/大学：Google Inc.
	- 模型：ResNet-50, ResNet-100, ResNet-150
	- 任务：ImageNet分类和COCO检测
	- 贡献：
		- 提供了一个在如Qualcomm Hexagon等纯整数算术硬件上高效实现的量化推理框架，并描述了一个在ARM NEON上高效、准确的实现。
		- 提供了一个与量化推理共同设计的量化训练框架，旨在最小化量化对真实模型精度的损失。

- #### Post-training piecewise linear quantization for deep neural networks (PWLQ) (2020) [[paper](https://openaccess.thecvf.com/content/ECCV2020/papers/PWLQ/PWLQ_Paper.pdf)]
	- 会议/期刊：ECCV, 2020
	- 大学/机构：1. Samsung Semiconductor, Inc., 2. Microsoft
	- 模型：Inception-v3, ResNet-50, MobileNet-v2
	- 背景：通过将神经网络从全精度转换为8位定点整数，训练后量化的统一方案取得了令人满意的结果。然而，当量化到较低位宽时，性能显著下降。
	- 精度显著下降的原因：预训练DNN的权重和激活分布呈钟形（高斯或拉普拉斯分布）。大多数权重聚集在零附近，很少有权重在长尾中传播。因此，当使用低位宽时，均匀量化会给小数量级分配过少的量化等级，给大数量级分配过多的量化等级，导致显著的精度下降。
	- 贡献：
		1. 提出了一种名为分段线性量化（Piecewise Linear Quantization，PWLQ）的方案，用于高效部署预训练的DNN，无需重新训练或访问完整的训练数据集。
		2. 提出了寻找最优断点的解决方案，并证明我们的方法比统一方案获得了更低的量化误差。


[[Back to Overview](#overview)]

### Other Tasks
#### Object Detection
- #### Improving Post-Training Quantization on Object Detection with Task Loss-Guided Lp Metric, arXiv, 2023. [[paper](https://arxiv.org/abs/2304.09785)]


#### Super Resolution
- #### Toward Accurate Post-Training Quantization for Image Super Resolution", CVPR, 2023. [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Tu_Toward_Accurate_Post-Training_Quantization_for_Image_Super_Resolution_CVPR_2023_paper.pdf)] [[code]( https://github.com/huawei-noah/Efficient-Computing/tree/master/Quantization/PTQ4SR)]

#### Point Cloud
- #### LiDAR-PTQ: Post-Training Quantization for Point Cloud 3D Object Detection, arXiv, 2023. [[paper](http://arxiv.org/abs/2401.15865)]  


[[Back to Overview](#overview)]



---

## References
* Online Resources:
    * [MQBench (Benchmark)](http://mqbench.tech/)
    * [Awesome Model Quantization (GitHub)](https://github.com/htqin/awesome-model-quantization)
    * [Awesome Transformer Attention (GitHub)](https://github.com/cmhungsteve/Awesome-Transformer-Attention)
