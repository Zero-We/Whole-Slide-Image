Whole-Slide-Image Analysis
===========================================
# Introduction
-------------------------------------------
**病理乃医学之本**，是疾病诊断的金标准，尤其对癌症的诊断、治疗和预后有着重要的意义。

传统病理检查的流程是：首先，通过术前穿刺或术中切除等方式从患者体内取出病变组织。然后，通过石蜡固定、切片、染色等步骤制成玻片，再由病理医生使用显微镜观察进行诊断。这种诊断方式给医生带来了繁重的工作量，在持续高压的工作环境下也增加了漏诊、误诊的可能性。另外，我国病理医生数量存在巨大缺口，病理资源分布极其不均衡，难以满足日益增加的临床诊断需求。

![pathology-procedure](https://github.com/Zero-We/Whole-Slide-Image/blob/main/image/pathology-procedure.png)

近年来，随着计算病理学的兴起和发展，这些问题都有望得到解决。计算病理学是指通过数字化技术将玻片扫描成数字病理全切片图像，再利用计算机技术进行分析以辅助病理医生做出决策的一种新型诊断流程。因此，主要问题在于：1.如何快速并准确地完成图像扫描；2.如何更加有效地辅助医生进行分析诊断。因此，如何利用AI技术或者计算机视觉的方法对数字病理全切片图像进行深入的分析和理解受到了越来越多研究者的关注。

但是，计算机辅助数字病理全切片图像分析还面临着一些挑战：

1. 数字病理全切片图像具有**超高的分辨率**（一般80,000x80,000以上），如果像普通图像一样直接输入模型进行处理会带来巨大的显存开销。
2. 病理图像有着**高度的异质性**，同属于一种疾病的病理图像所展示的细胞形态结构会有很大的差别。由于类内差异性很大，在训练数据较少的时候，模型很容易发生过拟合现象。
3. 全切片病理图像的**标注很难获取**，尤其是像细胞核实例分割任务，需要勾勒出所有细胞核的轮廓，工作量极其庞大。

# Classification
-------------------------------------------
WSI分类，包括疾病的诊断、癌症分级和分型等等。WSI分类的基本方法主要可以分为三大类：

1. 全监督方法
2. 弱监督方法
3. 混合监督方法


### 全监督方法  
由于WSI的像素级标注获取成本高，使用全监督学习对WSI分类的方法并不多，主要集中在Camelyon竞赛系列方法。虽然这一类方法成本较高，但在实际设计应用平台的过程中，全监督还是很重要的一步，因为它能够得到比较好的性能。  
</br>
</br>
#### Deep Learning for Identifying Metastatic Breast Cancer  
* 简介：Camelyon16竞赛的冠军方案，在比赛中能在Camelyon16测试集达到0.925AUC，正式比赛结束后重新提交的结果能达到0.9935AUC。  
* 方法：在tumor WSI中采样tumor patch和normal patch，在normal WSI中采样normal patch，从而训练patch classifier。为了进一步提高classifier的性能会重新采样hard negative patch用于训练。用训练好的patch classifier滑窗法产生每张WSI的heatmap， 从中提取出28个hand-crafted feature（如肿瘤区域面积）。 将WSI的特征输入至Random Forest classifier中进行分类。  
* [paper](https://arxiv.org/pdf/1606.05718.pdf)   

### 弱监督方法
由于WSI图像超高分辨率，无法直接使用普通的CNN模型进行处理，常见做法是将WSI图像划分为许多patch处理，patch是能够输入至普通的CNN网络中的。但是patch的标注成本过高，而WSI的图像级别标签是能够较容易从诊断报告中获取的。所以WSI分类问题实际上可以转化为仅利用WSI图像级别标签对patch进行分析从而完成对WSI分类的弱监督学习问题。其中一类最常见的做法，是借助多示例学习（Multiple Instance Learning）来解决。  
</br>
</br>
#### Clinical-grade Computational Pathology Using Weakly Supervised Deep Learning on Whole Slide Images
* MIL-RNN
* 简介：19年Nature Medicine，在WSI分类领域最基础的工作，被大量引用及验证。在超大规模数据集（超过40,000WSI）训练WSI分类模型，在三个不同癌种的数据集上都达到超过0.98AUC。使用MAX-MIL选择出patch再用RNN模型融合patch特征进行分类，其实很大部分性能来自于MAX-MIL。
* 方法：所有patch输入ResNet34中进行一次前传，选择分类分数最高的、或者top-k，记下它们的索引。重新取这部分patch进行前传和反向传播，训练MAX-MIL。用MAX-MIL对所有patch预测出一个分数，再选出分数最高的top-k，把它们的feature输入至RNN网络中训练，融合出WSI feature再进行分类。
*  [paper](https://www.nature.com/articles/s41591-019-0508-1)
*  [code](https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019) [官方代码]  
<img src="https://github.com/Zero-We/Whole-Slide-Image/blob/main/image/rnn-mil.png" width="800px">

#### Attention-based Deep Multiple Instance Learning  
* AbMIL
* 简介：AbMIL是WSI分类中经常被用来比较的方法。它是典型的ES范式的MIL方法，也就是通过先融合得到bag feature，再对bag进行分类。原文中虽然也在病理图像上测试，但不是WSI图像，而是分辨率小一点的patch。  
* 方法：简单说就是将instance feature输入至一个两层全连接网络，预测出每一个instance的attention weight，再用这个weight对instance feature进行加权和，得到bag feature从而进行分类。所以，不可能将WSI所有的patch输入至AbMIL中进行end-to-end的学习，两种方式。第一，采用类似MIL-RNN的方式，先预测出patch的分数，从中选择一部分，输入到Attention网络中进行融合。第二，使用预训练的特征提取器对所有patch进行编码，用patch feature直接训练Attention网络，但feature extractor是固定的。
*  [paper](http://proceedings.mlr.press/v80/ilse18a/ilse18a.pdf)  
*  [code](https://github.com/AMLab-Amsterdam/AttentionDeepMIL)[官方代码]
<img src="https://github.com/Zero-We/Whole-Slide-Image/blob/main/image/attention-mil.png" width="800px">

#### Dual-stream multiple instance learning network for whole slide image classification with self-supervised contrastive learning
* DSMIL  
* 简介：DSMIL比较创新性地使用了simclr自监督学习的方法预训练patch特征提取器，部分研究结果也表明自监督学习在WSI分析任务中确实能提高性能。此外，DSMIL使用了patch-classfier + wsi classifier jointly learning的方式进行学习，这种联合训练的方式也在一些工作中被提及。  
* 方法：使用simclr self-supervised learning 训练patch特征提取器，将patch提取为特征向量。然后采用dual-stream结构，第一个stream类似 MAX-Pooling MIL，使用一个FC求出所有patch的分类分数，从中锁定最重要的一个patch，作为patch-level输出。另一个stream，计算其它patch与key patch之间的similarity，从而得到每个patch的attention weight，对patch features进行线性组合得到WSI feature，得到wsi-level输出。结合了MAX-MIL与AbMIL的特点。  
*  [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Dual-Stream_Multiple_Instance_Learning_Network_for_Whole_Slide_Image_Classification_CVPR_2021_paper.pdf)  
*  [code](https://github.com/binli123/dsmil-wsi)[官方代码]  

#### TransMIL: Transformer based correlated multiple instance learning for whole slide image classification
* TransMIL
* 简介：TransMIL应该是WSI分类中首次引入Transformer网络结构的一个方法。传统的MIL方法都是基于instance相互独立假设，作者希望在此基础之上考虑同一个bag中不同instance之间的correlation，因此想到了使用self-attention机制对patch features进行融合。
* 方法：通过一个ImageNet预训练的Resnet-50网络将patch编码为feature vector，然后将一张WSI所有的patch feature输入至Transformer网络中。作者对Transformer网络进行了一些改进，首先去除了MLP部分，用他们自己设计的一个PPEG模块进行替代（这是一个卷积结构）；其次，作者使用Nystrom attention近似原有的self-attention计算，该方法能够大幅减少QKV计算过程的时间和空间复杂度。考虑到WSI所含patch数量众多，远多于普通自然图像中的token数量，该近似是必要且有效的，如若不近似便要采用聚类或采样等方式减少patch的数量。
* [paper](https://arxiv.org/pdf/2106.00908.pdf)
* [code](https://github.com/szc19990412/TransMIL)[官方代码]  
<img src="https://github.com/Zero-We/Whole-Slide-Image/blob/main/image/transmil.png" width="800px">

#### Data-efficient and weakly supervised computational pathology on whole-slide images
* CLAM
* 简介：21年Nature Biomedical Engineering的工作，个人觉得是继19年RNN-MIL之后一个很solid的工作，TransMIL很多实验细节也follow了这篇工作，特别是它处理WSI分类的整个pipeline是值得学习。它开源的代码涉及到了WSI的预处理，patch提取，patch encoding，model training，model evaluating以及可视化，非常全面，能够学习到很多。
* 方法：通过一个ImageNet预训练的Resnet-50网络将patch编码为feature vector，输入至CLAM网络中进行分类。作者设计的CLAM网络实际上非常类似于attention-mil，用一个fc层进行特征变换后直接接attention net计算各个feature的attention weight然后线性组合为WSI feature进行分类。不同点在于作者提出了一个instance-level clustering，从代码层面理解这个模块，其实是它在WSI分类上引入了一个patch分类的分支（目前有好几个方法都采用了这种joint learning的方式），根据attention weight排序选择靠前和靠后的8-10个，赋予两类标签（是该类或者背景）。除此之外，作者提出使用mulit-branch的方式处理多类MIL的问题。
* [paper](https://www.nature.com/articles/s41551-020-00682-w)
* [code](https://github.com/mahmoodlab/CLAM)[官方代码]
<img src="https://github.com/Zero-We/Whole-Slide-Image/blob/main/image/clam.png" width="800px">


# Segmentation
-------------------------------------------
病理图像分割：区域分割、细胞核分割等等。涉及TME(Tumor Micro Environment)的分析，对生物标记物(bio-marker)的发现和制药有着重要的意义。同时，根据细胞核分割结果，获取肿瘤细胞、正常细胞的形态特征能够进一步用于WSI图像分类，并且有着较好的可解释性。目前，组织病理图像分割主要包含两方面的工作：  

1. 细胞核分割  
2. 腺体分割  

## 细胞核分割  
目前，病理图像细胞核分割方面已经有较多的工作，这方面的研究比WSI分类还要来得更早，开源病理分析软件譬如CellProfiler和QuPath都集成了很好的分割算法，早期的工作主要是借助传统视觉方法像active contour, watershed, level set来完成，15年后开始转向深度学习。目前的细胞核分割算法主要还是以全监督方法为主，考虑到细胞核分割数据集标注十分困难，还有一些其它方法研究弱监督（点标注）、无监督（自监督、迁移学习）、半监督等不同的设置。  

1. 全监督方法
2. 弱监督方法
3. 混合监督方法

### 全监督方法  
由于Neeraj Kumar等人的努力，细胞核分割方面开展了几次公开挑战赛，因此出现了几个标注完备的组织病理图像细胞核实例分割数据集。譬如Kumar以及它的拓展版MoNuSeg和MoNuSAC等，CoNSep，PanNude，Lizard，TNBC等等。这促进了大量全监督细胞核分割方法的发展，但是全监督方法毕竟还是要依赖于大量标注数据，这为它的实际应用带来了很大困难。  

#### Hover-net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images
* Hover-net
* 简介：Hover-net是全监督细胞核分割上具有很高影响力的工作，由Nasir Rajpoot等人完成，此人在细胞核分割方面有着非常多的工作。它的整体方案简洁有效 性能优异，且开源代码很规范、可读性高、易拓展。我们试过测试Hover-net迁移到不同数据集的性能，效果也是非常好的。另外，我个人还做半监督的细胞核分割，因此测试过不同标注样本下的分割性能，Hover-net在这方面也要优于很多其它方法。此外，该工作还公开了一个细胞核分割数据集CoNSeP，是一个非常充实的工作。    
* 方法：实际上Hover-net能够同时完成细胞核分割以及分类，这里着重介绍分割分支。细胞核分割与其它自然图像分割不一样，它的语义类别相对较少，但是一张图像上会有较多object，这也使得它们各自的处理方法大有不同。细胞核实例分割既可以采用mask r-cnn先检测再分割的框架，也可以在语义分割基础上区分不同的object。目前较多的工作都属于后面一类，Hover-net也不例外。语义分割与实例分割最大的区别在于touching object，如果所有的object都是分离的，那么两者是没有区别的，我们只需要在语义分割上做连通域标记就出来实例分割结果了。但如果oject之间相互粘连，我们一般是使用watershed做后处理，因此就需要一个distance map和marker。Hover-net就是通过网络来预测horizontal distance map和vertical distance map来辅助语义分割结果分离出不同的object。所谓的horizontal distance map就是细胞内每个像素到质心的水平距离，不过把它rescale到-1到1罢了，vertical distance map同理，所以它所采用的distance map就非常简单有效且易于学习。缺点是它的后处理非常依赖于一个阈值。  
*  [paper](https://www.sciencedirect.com/science/article/pii/S1361841519301045?via%3Dihub)  
*  [code](https://github.com/vqdang/hover_net)[官方代码]
<img src="https://github.com/Zero-We/Whole-Slide-Image/blob/main/image/hovernet.png" width="800px">

#### CIA-Net: Robust nuclei instance segmentation with contour-aware information aggregation
* CIA-Net
* 简介：19年MICCAI的工作，是Chen Hao等人完成的，他们在病理图像细胞核实例分割方面有比较多出色的工作，而他们解决细胞核分割一贯的思路是预测细胞核边缘。  
* 方法：CIA-Net采用的是全监督细胞核实例分割的另一种思路，不同于Hover-net通过学习细胞核实例的距离图分割粘连实例，CIA-Net以及后续介绍的DCAN等工作是通过细胞核的边缘来分割粘连实例。总体思路是同时预测细胞核语义分割结果，以及细胞核边缘，再用语义分割结果减去细胞核边缘将粘连实例相互分离。但实际上，由于细胞核边缘是不规则曲线，难以直接学习得到。因此，该工作通过引入IAM特征融合模块以及Smooth Truncated Loss，从而更好地学习细胞核边缘并且利用细胞核边缘信息实现更好的细胞核实例分割。  
*  [paper](https://link.springer.com/chapter/10.1007/978-3-030-20351-1_53)  
*  [code][暂无]  
<img src="https://github.com/Zero-We/Whole-Slide-Image/blob/main/image/cia-net.png" width="800px">

#### 

### 弱监督方法  

# Survival Prediction
-------------------------------------------
病理图像分析患者预后情况，包括预测五年生存期、十年生存期、以及预测精确的存活时间。目前，研究重点转向使用多模态融合（如病理+基因）以提高预后分析能力。利用病理图像预测患者预后风险也有不少的工作，早期是先让医生从WSI中挑选ROI区域，再根据ROI区域分析患者预后，近年的工作基本都是直接根据WSI图像预测预后风险。  




# Public Dataset for WSIs
-------------------------------------------

This is a link to TCGA website. <https://portal.gdc.cancer.gov/>

Link to Camelyon: <https://camelyon16.grand-challenge.org/>

Prospect
-------------------------------------------
