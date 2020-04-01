

Unsupervised Few-shot Learning via Distribution Shift-based Augmentation
---

### Introduction
This is the official implementation of 

[Unsupervised Few-shot Learning via Distribution Shift-based Augmentation](https://arxiv.org/abs/1910.08343)

Tiexin Qin, Wenbin Li, Yinghuan Shi and Yang Gao.


<center>
<img src="./figs/framework.png" width="90%" height="50%" />
</center>


### Abstract
Few-shot learning aims to learn a new concept when only a few training examples are available, which has been extensively explored in recent years. However, most of the current works heavily rely on a large-scale labeled auxiliary set to train their models in an episodic-training paradigm. Such a kind of supervised setting basically limits the widespread use of few-shot learning algorithms, especially in real-world applications. Instead, in this paper, we develop a novel framework called \emph{Unsupervised Few-shot Learning via Distribution Shift-based Data Augmentation} (ULDA), which pays attention to the distribution diversity inside each constructed pretext task when using data augmentation. Importantly, we highlight the value and importance of the distribution diversity in the augmentation-based pretext tasks. In ULDA, we systemically investigate the effects of different augmentation techniques and propose to strengthen the distribution diversity (or difference) between the query set and support set in each few-shot task, by augmenting these two sets separately (\ie~shifting). In this way, even incorporated with simple augmentation techniques (\eg~random crop, color jittering, or rotation), our ULDA can produce a significant improvement. In the experiments, the few-shot models learned by ULDA can achieve superior generalization performance and obtain state-of-the-art results in a variety of established few-shot learning tasks on \emph{mini}ImageNet and \emph{tiered}ImageNet. The source code is available in \textcolor{blue}{\emph{https://github.com/WonderSeven/ULDA}}.


### Demo
<video src="./figs/demo.mp4" width="520" height="400"
controls="controls"></video> 


### Citation    
    @inproceedings{Shi2019DeepAugNet,
    title={Automatic Data Augmentation by Learning the Deterministic Policy},
    author={Yinghuan Shi, Tiexin Qin, Yong Liu, Jiwen Lu,  Gao Yang and Dinggang Shen},
    booktitle={arxiv},
    year={2019}

