# Papers : Biological and Artificial Neural Networks
ニューラルネットワークの論文のなかで計算神経科学と関係しているものを集めました。読んだ論文しか掲載できていないので、他にもある場合はIssueからお願いします。

## ニューラルネットワークと計算神経科学

### 全般
- A.H. Marblestone, G. Wayne, K.P. Kording. Toward an integration of deep learning and neuroscience. 2016. Front. Comput. Neurosci., 10, p.94
https://www.frontiersin.org/articles/10.3389/fncom.2016.00094/full

(参考)Artificial Neural Networks as Models of Neural Information Processing
https://www.frontiersin.org/research-topics/4817/artificial-neural-networks-as-models-of-neural-information-processing

- D. Cox, T. Dean. Neural networks and neuroscience-inspired computer vision. Curr Biol. 2014. doi: 10.1016/j.cub.2014.08.026.

### 計算神経科学の中のニューラルネットワーク
計算神経科学の総説の中でニューラルネットワークに触れているもの。

### ニューラルネットワークの解析手法
ニューラルネットワークの神経表現を理解するための手法。
- D. Barrett, A. Morcos, J. Macke. Analyzing biological and artificial neural networks: challenges with opportunities for synergy?. 2018. https://arxiv.org/abs/1810.13373
https://arxiv.org/abs/1706.05806

### その他
- D. Silva, P. Cruz, A. Gutierrez. Are the long-short term memory and convolution neural net biological system?. 2018. doi: 10.1016/j.icte.2018.04.001

## ニューラルネットによる脳内の神経表現の再現
脳の神経表現の理解は難しい。ニューラルネットワークに特定のタスクを学習（特定の損失関数に対して最適化）させると、脳の神経表現と同じ表現を獲得する場合がある。このとき、間接的に脳の神経表現の目的を知ることができる(Whyの解決手法)。

・vector-based navigation
・https://www.biorxiv.org/content/biorxiv/early/2018/01/05/242867.full.pdf

・Performance-optimized hierarchical models predict neural responses in higher visual cortex
・Deep Supervised, but Not Unsupervised, Models May Explain IT Cortical Representation
・The emergence of multiple retinal cell types through efficient coding of natural movies
・https://arxiv.org/pdf/1811.02290.pdf
・https://arxiv.org/pdf/1810.11594.pdf
https://www.biorxiv.org/content/biorxiv/early/2018/07/30/380048.full.pdf

Brain-Score:
https://www.biorxiv.org/content/biorxiv/early/2018/09/05/407007.full.pdf


https://www.biorxiv.org/content/biorxiv/early/2018/05/02/133694.full.pdf

E. Kim, D. Hannan, G. Kenyon. Deep Sparse Coding for Invariant Multimodal Halle Berry Neurons. 2018. CVPR. https://arxiv.org/abs/1711.07998

時間
https://www.biorxiv.org/content/biorxiv/early/2017/04/10/125849.full.pdf

### 視覚

### 場所受容野

### おばあさん細胞

### 運動野

## 神経科学に基づいたアーキテクチャ

###　全般
・Neuroscience-Inspired Artificial Intelligence
https://www.cell.com/neuron/fulltext/S0896-6273(17)30509-3
https://www.sciencedirect.com/science/article/pii/S0896627317305093

### PredNet
- W. Lotter et al. Deep predictive coding networks for video prediction and unsupervised learning. 2017. ICLR.

- W. Lotter et al. A neural network trained to predict future video frames mimics critical properties of biological neuronal responses and perception. 2018. CoRR abs/1805.10734 [arxiv]

- E. Watanabe et al. Illusory Motion Reproduced by Deep Neural Networks Trained for Prediction. Frontiers in Psychology. 2018;9:345. doi:10.3389/fpsyg.2018.00345. [PubMed]
ヘビの回転錯視がPredNetでも生じたという内容。

### その他のアーキテクチャ
- subLSTM:Cortical microcircuits as gated-recurrent neural networks

・lateral inhibition inspired cnn for visual

https://www.biorxiv.org/content/biorxiv/early/2018/04/10/268375.full.pdf

## 生物学的妥当性のある学習法則
- B. Scellier, Y. Bengio. Towards a biologically plausible backprop. 2016. arXiv:1602.05179 914

https://www.biorxiv.org/content/biorxiv/early/2018/08/29/390435.full.pdf

多層の線形NNはSGDで学習した場合、特異値分解を獲得し、大きい特異値に対応する成分ほど速く学習し、階層的な成分は学習中に段階的に分化していく。これらの現象は人の認知発達でみられる現象と一致し、最適解が唯一であるため異なる人で同じ意味構造を持つことも説明できる
https://arxiv.org/abs/1810.10531

時系列問題の学習には時間の逆方向に誤差を伝搬させるBPTTが使われているが、長距離の依存は扱えず、脳内でも実現不可能そうである。現在の状態から少数の過去の状態を思い出し、それらに対し注意機構としてスキップ接続を張ることで長距離の信用割当問題を解ける https://arxiv.org/abs/1809.03702

脳内でもマイクロ回路を適切に構成することで、誤差逆伝搬法と同じ学習を実現でき、さらに順伝搬と逆伝搬の２フェイズを分ける必要がなく連続的に計算できる。他の学習則は非力であり（https://arxiv.org/abs/1807.04587 ）、脳内で誤差逆伝搬法が使われている可能性がある　 https://arxiv.org/abs/1810.11393

## その他
https://arxiv.org/abs/1502.04156
https://www.nature.com/articles/ncomms13276
https://arxiv.org/abs/1807.11819

ニューラルネットワークの発達
https://www.biorxiv.org/content/early/2018/08/09/374439
