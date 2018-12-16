# Papers : Biological and Artificial Neural Networks
ニューラルネットワークの論文のなかで計算神経科学と関係しているもの（の中で個人的に気になったもの）を集めました。重要なのに記載できてない論文がある場合はIssueからお願いします。

## ニューラルネットワークと計算神経科学

### 全般
- A.H. Marblestone, G. Wayne, K.P. Kording. Toward an integration of deep learning and neuroscience. 2016. *Front. Comput. Neurosci.*, 10, p.94
https://www.frontiersin.org/articles/10.3389/fncom.2016.00094/full

(cf.)Artificial Neural Networks as Models of Neural Information Processing
https://www.frontiersin.org/research-topics/4817/artificial-neural-networks-as-models-of-neural-information-processing

- D. Cox, T. Dean. Neural networks and neuroscience-inspired computer vision. Curr Biol. 2014. doi: 10.1016/j.cub.2014.08.026.

### 計算神経科学の中のニューラルネットワーク
計算神経科学の総説の中でニューラルネットワークに触れているもの。

### ニューラルネットワークの解析手法
ニューラルネットワークの神経表現を理解するための手法。Saliency mapとかは触れません。
- D. Barrett, A. Morcos, J. Macke. "Analyzing biological and artificial neural networks: challenges with opportunities for synergy?". 2018. ([arxiv](https://arxiv.org/abs/1810.13373))

#### SVCCA
- M. Raghu, J. Gilmer, J. Yosinski, J. Sohl-Dickstein. "SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability". *NIPS* (2017). ([arxiv](https://arxiv.org/abs/1706.05806))

### その他
- D. Silva, P. Cruz, A. Gutierrez. Are the long-short term memory and convolution neural net biological system?. 2018. doi: 10.1016/j.icte.2018.04.001

## ニューラルネットによる脳の神経表現の再現
脳の神経表現の理解は難しい。ニューラルネットワークに特定のタスクを学習（特定の損失関数に対して最適化）させると、脳の神経表現と同じ表現を獲得する場合がある。このとき、間接的に脳の神経表現の目的を知ることができる(Whyの解決手法)。

- https://www.biorxiv.org/content/biorxiv/early/2018/01/05/242867.full.pdf

- Performance-optimized hierarchical models predict neural responses in higher visual cortex
- Deep Supervised, but Not Unsupervised, Models May Explain IT Cortical Representation
- The emergence of multiple retinal cell types through efficient coding of natural movies
- https://arxiv.org/pdf/1811.02290.pdf
- https://arxiv.org/pdf/1810.11594.pdf
- https://www.biorxiv.org/content/biorxiv/early/2018/07/30/380048.full.pdf

- https://www.biorxiv.org/content/biorxiv/early/2018/05/02/133694.full.pdf


### 視覚
ニューラルネットワークが一番成功したといってもいいのが視覚。そのため論文の数も多い。

#### Brain-Score
https://www.biorxiv.org/content/biorxiv/early/2018/09/05/407007.full.pdf

### 運動野
- D. Sussillo, M. Churchland, M. Kaufman, K. Shenoy. "A neural network that finds a naturalistic solution for the production of muscle activity". *Nat. Neurosci.* **18**(7), 1025–1033 (2015). ([PubMed](https://www.ncbi.nlm.nih.gov/pubmed/26075643))

### 場所受容野
- C. Cueva, X. Wei. "Emergence of grid-like representations by training recurrent neural networks to perform spatial localization". ICLR (2018). ([arxiv](https://arxiv.org/abs/1803.07770))
- A. Banino, et al. "Vector-based navigation using grid-like representations in artificial agents", *Nat.* **557**(7705), 429–433 (2018). ([pdf](https://deepmind.com/documents/201/Vector-based%20Navigation%20using%20Grid-like%20Representations%20in%20Artificial%20Agents.pdf))

### おばあさん細胞
- E. Kim, D. Hannan, G. Kenyon. Deep Sparse Coding for Invariant Multimodal Halle Berry Neurons. *CVPR* (2018). ([arxiv](https://arxiv.org/abs/1711.07998))

## 神経科学に基づいたアーキテクチャ

###　全般
- D. Hassabis, D. Kumaran, C. Summerfield, M. Botvinick. "Neuroscience-Inspired Artificial Intelligence". *Neuron* **95**(2), 245-258 (2017).
([sciencedirect](https://www.sciencedirect.com/science/article/pii/S0896627317305093))

### PredNet
- W. Lotter et al. "Deep predictive coding networks for video prediction and unsupervised learning". *ICLR* (2017).

- W. Lotter et al. "A neural network trained to predict future video frames mimics critical properties of biological neuronal responses and perception". 2018. *CoRR* abs/1805.10734 [arxiv]  
- E. Watanabe, A. Kitaoka, K. Sakamoto, M. Yasugi, K. Tanaka. "Illusory Motion Reproduced by Deep Neural Networks Trained for Prediction". *Front. Psychol.* (2018). ([Front. Psychol.](https://www.frontiersin.org/articles/10.3389/fpsyg.2018.00345/full))

### その他のアーキテクチャ
- subLSTM:Cortical microcircuits as gated-recurrent neural networks

- lateral inhibition inspired cnn for visual
側方抑制をCNNに使ってみたというもの。

https://www.biorxiv.org/content/biorxiv/early/2018/04/10/268375.full.pdf

## 学習と発達
### 生物学的妥当性のある学習法則
ニューラルネットワークの強力な学習アルゴリズムである誤差逆伝搬法(Back propagation)は生物学的に妥当である(biological plausible)とは言えない。そこで、生体内でも可能と言えそうな学習方法が考案されている。
- Yoshua Bengio, Dong-Hyun Lee, J. Bornschein, T. Mesnard, Z. Lin. Towards Biologically Plausible Deep Learning
- B. Scellier, Y. Bengio. Towards a biologically plausible backprop. 2016. arXiv:1602.05179 914

- Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation
https://www.biorxiv.org/content/biorxiv/early/2018/08/29/390435.full.pdf


- N. Ke, A. Goyal, O. Bilaniuk, J. Binas, M. Mozer, C. Pal, Y. Bengio. "Sparse Attentive Backtracking: Temporal CreditAssignment Through Reminding". 2018. NIPS. ([arxiv](https://arxiv.org/abs/1809.03702))
- S. Bartunov, A. Santoro, B. Richards, L. Marris, G. Hinton, T. Lillicrap. "Assessing the Scalability of Biologically-Motivated Deep Learning Algorithms and Architectures". 2018.  NIPS. ([arxiv](https://arxiv.org/abs/1807.04587))
- J. Sacramento, R. P. Costa, Y. Bengio, W. Senn. "Dendritic cortical microcircuits approximate the backpropagation algorithm". 2018. NIPS. ([arxiv](https://arxiv.org/abs/1810.11393))

### ニューラルネットワークと脳の発達
「発達」というのは技術的発達ではなく、幼児の脳が如何にして成人と同じような脳機能を獲得するか、ということの発達。
- A. M. Saxe, J. L. McClelland, S. Ganguli. "A mathematical theory of semantic development in deep neural networks". 2018. ([arxiv](https://arxiv.org/abs/1810.10531))
- J. Shen, M. D. Petkova, F. Liu, C. Tang. "Toward deciphering developmental patterning with deep neural network". 2018. ([bioRxiv](https://www.biorxiv.org/content/early/2018/08/09/374439))

## その他
https://arxiv.org/abs/1502.04156
https://www.nature.com/articles/ncomms13276
https://arxiv.org/abs/1807.11819
