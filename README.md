# A Survey of Deep Learning for Time Series Forecasting

<div style="text-align: center;">
  <img src="./timeline.jpg" alt="image info">
</div>

🚩 2017: Recurrent neural networks ([RNNs](https://www.sciencedirect.com/science/article/abs/pii/036402139090002E)), such as [DA-RNN](https://www.ijcai.org/proceedings/2017/0366.pdf) and [MQRNN](https://arxiv.org/pdf/1711.11053), emerged as the dominant approach, marking the beginning of rapid advancement of deep learning in TSF.

🚩 2018: Data in various fields exhibit both spatial and temporal dependencies, and graph neural networks ([GNNs](https://ieeexplore.ieee.org/document/4700287)) have introduced novel perspectives for spatiotemporal modeling, with [STGCN](https://www.ijcai.org/proceedings/2018/0505.pdf) widely adopted as a benchmark. 

🚩 2019: [Transformer](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)-based models , such as [LogTrans](https://proceedings.neurips.cc/paper_files/paper/2019/file/6775a0635c302542da2c32aa19d86be0-Paper.pdf) and [TFT](https://www.sciencedirect.com/science/article/pii/S0169207021000637), have gained popularity due to their strength in modeling global dependencies. Concurrently, convolutional neural networks ([CNNs](https://aclanthology.org/D14-1181/)) (e.g., [DeepGLO](https://proceedings.neurips.cc/paper_files/paper/2019/file/3a0844cee4fcf57de0c71e9ad3035478-Paper.pdf), [MICN](https://openreview.net/forum?id=zt53IDUR1U)) were employed in TSF, leveraging their parallelism, parameter sharing, and local perception capabilities. 

🚩 2020: Multi-layer perceptrons ([MLPs](https://psycnet.apa.org/record/1959-09865-001)), due to their simple architecture and ease of implementation, have been widely applied in TSF. A representative example is [N-BEATS](https://arxiv.org/abs/1905.10437), which inspired a series of follow-up variants such as [NBEATSx](https://www.sciencedirect.com/science/article/pii/S0169207022000413) and [NHiTS](https://ojs.aaai.org/index.php/AAAI/article/view/25854).

🚩 2021: Given that time series forecasting can essentially be regarded as a generative task, some generative approaches, such as [TimeGrad](https://proceedings.mlr.press/v139/rasul21a/rasul21a.pdf) and [MAF](https://openreview.net/forum?id=WiGQBFuVRv), model the underlying data distribution to generate future sequences. 

🚩 2022: The Transformer architecture has undergone continuous development in recent years, giving rise to numerous studies, including [FEDformer](https://proceedings.mlr.press/v162/zhou22g/zhou22g.pdf) and [Pyraformer](https://openreview.net/forum?id=0EXmFzUn5I). 

🚩 2023: [DLinear](https://ojs.aaai.org/index.php/AAAI/article/view/26317), a model based purely on MLPs, argued that Transformers are not necessarily superior in TSF and asserted that linear neural networks might be equally effective. 

🚩 2024: [iTransformer](https://openreview.net/forum?id=JePfAI8fah) demonstrated the effectiveness of the Transformer architecture through structural optimization, providing a strong rebuttal to DLinear's claims. Furthermore, the rapid proliferation of methods based on large language models (LLMs), such as [Time-LLM](https://openreview.net/forum?id=Unb5CVPtae), provides further evidence for the feasibility of Transformers in TSF.

<div style="text-align: center;">
  <img src="./taxonomy.jpg" alt="taxonomy">
</div>

📍 We provide a systematic review of deep learning-based TSF methods, summarizing recent advancements.  Specifically, we propose **a novel taxonomy based on core modeling paradigms**, which categorizes existing methods into three paradigms: **discriminative, generative, and plug-and-play**. Additionally, we summarize commonly used datasets and evaluation metrics, and discuss current challenges and future research directions in this field.

🚀 For a deeper dive, please check out our survey paper: **A Survey of Deep Learning for Time Series Forecasting: Taxonomy, Analysis, and Future Directions** 

## 📑 Table of Contents
- 🌟[A Survey of Deep Learning for Time Series Forecasting](#a-survey-of-deep-learning-for-time-series-forecasting)
  - 📑[Table of Contents](#-table-of-contents)
  - 📖[Taxonomy](#-taxonomy)
    - 📚[Discriminative Paradigm](#-discriminative-paradigm)
      - 🌟[MLP-based Methods](#mlp-based-methods)
      - 🌟[CNN-based Methods](#cnn-based-methods)
        - [CNN](#cnn)
        - [TCN](#tcn)
      - 🌟[RNN-based Methods](#rnn-based-methods)
        - [RNN](#rnn)
        - [GRU / LSTM](#gru--lstm)
      - 🌟[GNN-based Methods](#gnn-based-methods)
      - 🌟[Transformer-based Methods](#transformer-based-methods)
        - [Transformer](#transformer)
        - [Discriminative LLM](#discriminative-llm)
      - 🌟[Compound Model-based Methods](#compound-model-based-methods)
        - [CNN + RNN](#cnn--rnn)
        - [CNN + Transormer](#cnn--transformer)
        - [GNN + RNN](#gnn--rnn)
        - [GNN + Transormer](#gnn--transformer)
    - 📚[Generative Paradigm](#-generative-paradigm)
      - 🌟[Generative Model-based Methods](#generative-model-based-methods)
        - [GAN](#gan)
        - [VAE](#vae)
        - [Flow-based models](#flow-based-models)
        - [Diffusion models](#diffusion-models)
      - 🌟[Generative LLM-based Methods](#generative-llm-based-methods)
    - 📚[Plug-and-play Paradigm](#-plug-and-play-paradigm)

##  📖 Taxonomy
### 📚 Discriminative Paradigm

#### 🌟MLP-based Methods
- N-BEATS: Neural basis expansion analysis for interpretable time series forecasting, ICLR 2020. [[paper](https://arxiv.org/abs/1905.10437)] [[code](https://github.com/philipperemy/n-beats)]
- Neural basis expansion analysis with exogenous variables: Forecasting electricity prices with NBEATSx, IJoF 2022. [[paper](https://www.sciencedirect.com/science/article/pii/S0169207022000413)] [[code](https://github.com/cchallu/nbeatsx)]
- Nhits: Neural hierarchical interpolation for time series forecasting, AAAI 2023. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25854)] [[code](https://github.com/cchallu/n-hits)]
- Are transformers effective for time series forecasting?, AAAI 2023. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/26317)] [[code](https://github.com/cure-lab/LTSF-Linear)]
- Frequency-domain MLPs are more effective learners in time series forecasting, NIPS 2023. [[paper](https://arxiv.org/abs/2311.06184)] [[code](https://github.com/aikunyi/FreTS)]
- Tsmixer: Lightweight mlp-mixer model for multivariate time series forecasting, KDD 2023. [[paper](https://arxiv.org/abs/2306.09364)]
- TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting, ICLR 2024. [[paper](https://openreview.net/forum?id=7oLshfEIC2)] [[code](https://github.com/kwuking/TimeMixer)]
- Hdmixer: Hierarchical dependency with extendable patch for multivariate time series forecasting, AAAI 2024. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/29155)] [[code](https://github.com/hqh0728/HDMixer)]
- Wpmixer: Efficient multi-resolution mixing for long-term time series forecasting, AAAI 2025. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/34156)] [[code](https://github.com/Secure-and-Intelligent-Systems-Lab/WPMixer)]

#### 🌟CNN-based Methods
##### CNN
- TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis, ICLR 2023. [[paper](https://arxiv.org/abs/2210.02186)] [[code](https://github.com/thuml/TimesNet)]
- Modeling temporal patterns with dilated convolutions for time-series forecasting, TKDD 2021. [[paper](https://dl.acm.org/doi/abs/10.1145/3453724)]
- Convtimenet: A deep hierarchical fully convolutional model for multivariate time series analysis, WWW 2025. [[paper](https://arxiv.org/abs/2403.01493)] [[code](https://github.com/Mingyue-Cheng/ConvTimeNet)]
- TVNet: A Novel Time Series Analysis Method Based on Dynamic Convolution and 3D-Variation, ICLR 2025. [[paper](https://openreview.net/forum?id=MZDdTzN6Cy)]
##### TCN
- Moderntcn: A modern pure convolution structure for general time series analysis, ICLR 2024. [[paper](https://openreview.net/forum?id=vpJMJerXHU)] [[code](https://github.com/luodhhh/ModernTCN)]
- Think globally, act locally: A deep neural network approach to high-dimensional time series forecasting, NIPS 2019. [[paper](https://arxiv.org/abs/1905.03806)] [[code](https://github.com/rajatsen91/deepglo)]
- Micn: Multi-scale local and global context modeling for long-term series forecasting, ICLR 2023. [[paper](https://openreview.net/forum?id=zt53IDUR1U)] [[code](https://github.com/wanghq21/MICN)]
- Scinet: Time series modeling and forecasting with sample convolution and interaction, NIPS 2022. [[paper](https://arxiv.org/abs/2106.09305)] [[code](https://github.com/cure-lab/SCINet)]
- Cross-LKTCN: Modern convolution utilizing cross-variable dependency for multivariate time series forecasting dependency for multivariate time series forecasting, Arxiv 2023. [[paper](https://arxiv.org/abs/2306.02326)]

#### 🌟RNN-based Methods
##### RNN
- A multi-horizon quantile recurrent forecaster, NIPSW 2017. [[paper](https://arxiv.org/abs/1711.11053)] [[code](https://github.com/tianchen101/MQRNN)]
- DeepAR: Probabilistic forecasting with autoregressive recurrent networks, IJoF 2019. [[paper](https://arxiv.org/abs/1704.04110)] [[code](https://github.com/brunoklein99/deepar)]
- Modeling irregular time series with continuous recurrent units, ICML 2022. [[paper](https://arxiv.org/abs/2111.11344)] [[code](https://github.com/boschresearch/continuous-recurrent-units)]
- Segrnn: Segment recurrent neural network for long-term time series forecasting, Arxiv 2023. [[paper](https://arxiv.org/abs/2308.11200)] [[code](https://github.com/lss-1138/SegRNN)]
##### GRU / LSTM
- Addressing Prediction Delays in Time Series Forecasting: A Continuous GRU Approach with Derivative Regularization, KDD 2024. [[paper](https://arxiv.org/abs/2407.01622)] [[code](https://github.com/sheoyon-jhin/CONTIME)]
- A hybrid method of exponential smoothing and recurrent neural networks for time series forecasting, IJoF 2020. [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0169207019301153)]
- xlstmtime: Long-term time series forecasting with xlstm, AI 2024. [[paper](https://www.mdpi.com/2673-2688/5/3/71)] [[code](https://github.com/muslehal/xLSTMTime)]
- Unlocking the power of lstm for long term time series forecasting, AAAI 2025. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/33303)] [[code](https://github.com/Eleanorkong/P-sLSTM)]

#### 🌟GNN-based Methods
- Spectral temporal graph neural network for multivariate time-series forecasting, NIPS 2020. [[paper](https://arxiv.org/abs/2103.07719)] [[code](https://github.com/microsoft/StemGNN)]
- Multivariate time-series forecasting with temporal polynomial graph neural networks, NIPS 2022. [[paper](https://openreview.net/forum?id=pMumil2EJh)] [[code](https://github.com/zyplanet/TPGNN)]
- FourierGNN: Rethinking multivariate time series forecasting from a pure graph perspective, NIPS 2023. [[paper](https://arxiv.org/abs/2311.06190)] [[code](https://github.com/aikunyi/FourierGNN)]
- Connecting the dots: Multivariate time series forecasting with graph neural networks, KDD 2020. [[paper](https://arxiv.org/abs/2005.01165)] [[code](https://github.com/nnzhan/MTGNN)]
- Biased temporal convolution graph network for time series forecasting with missing values, ICLR 2024. [[paper](https://openreview.net/forum?id=O9nZCwdGcG)] [[code](https://github.com/chenxiaodanhit/BiTGraph)]
- Multi-scale adaptive graph neural network for multivariate time series forecasting, TKDE 2023. [[paper](https://ieeexplore.ieee.org/abstract/document/10105527)] [[code](https://github.com/shangzongjiang/MAGNN)]
- Graph-based Time Series Clustering for End-to-End Hierarchical Forecasting, ICML 2024. [[paper](https://arxiv.org/abs/2305.19183)] [[code](https://github.com/andreacini/higp)]
- Msgnet: Learning multi-scale inter-series correlations for multivariate time series forecasting, AAAI 2024. [[paper](https://arxiv.org/abs/2401.00423)] [[code](https://github.com/YoZhibo/MSGNet)]
- TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series Forecasting, ICML 2025. [[paper](https://arxiv.org/abs/2501.13041)] [[code](https://github.com/troubadour000/timefilter)]

#### 🌟Transformer-based Methods
##### Transformer
- A Time Series is Worth 64 Words: Long-term Forecasting with Transformers, ICLR 2023. [[paper](https://arxiv.org/abs/2211.14730)] [[code](https://github.com/yuqinie98/patchtst)]
- Non-stationary Transformers: Rethinking the Stationarity in Time Series Forecasting, NIPS 2022. [[paper](https://arxiv.org/abs/2205.14415)] [[code](https://github.com/thuml/Nonstationary_Transformers)]
- Scalable Transformer for High Dimensional Multivariate Time Series Forecasting, CIKM 2024. [[paper](https://arxiv.org/abs/2408.04245)] [[code](https://github.com/xinzzzhou/ScalableTransformer4HighDimensionMTSF)]
- VarDrop: Enhancing Training Efficiency by Reducing Variate Redundancy in Periodic Time Series Forecasting, AAAI 2025. [[paper](https://arxiv.org/abs/2501.14183)] [[code](https://github.com/kaist-dmlab/VarDrop)]
- iTransformer: Inverted transformers are effective for time series forecasting, ICLR 2024. [[paper](https://arxiv.org/abs/2310.06625)] [[code](https://github.com/thuml/iTransformer)]
- Pathformer: Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting, ICLR 2024. [[paper](https://openreview.net/forum?id=lJkOCMP2aW)] [[code](https://github.com/decisionintelligence/pathformer)]
- Informer: Beyond efficient transformer for long sequence time-series forecasting, AAAI 2021. [[paper](https://arxiv.org/abs/2012.07436)] [[code](https://github.com/zhouhaoyi/Informer2020)]
- Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting, NIPS 2021. [[paper](https://arxiv.org/abs/2106.13008)] [[code](https://github.com/thuml/Autoformer)]
- Fedformer: Frequency enhanced decomposed transformer for long-term series forecasting, ICML 2022. [[paper](https://arxiv.org/abs/2201.12740)] [[code](https://github.com/MAZiqing/FEDformer)]
- Adversarial sparse transformer for time series forecasting,	NIPS 2020. [[paper](https://proceedings.neurips.cc/paper/2020/file/c6b8c8d762da15fa8dbbdfb6baf9e260-Paper.pdf)] [[code](https://github.com/hihihihiwsf/AST)]
- Preformer: predictive transformer with multi-scale segment-wise correlations for long-term time series forecasting, ICASSP 2023. [[paper](https://arxiv.org/abs/2202.11356)] [[code](https://github.com/ddz16/Preformer)]
- Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and Forecasting, ICLR 2022. [[paper](https://openreview.net/pdf?id=0EXmFzUn5I)] [[code](https://github.com/ant-research/Pyraformer)]
- Contiformer: Continuous-time transformer for irregular time series modeling, NIPS 2023. [[paper](https://seqml.github.io/contiformer/)] [[code](https://github.com/microsoft/SeqML/tree/main/ContiFormer)]
- Fredformer: Frequency debiased transformer for time series forecasting, KDD 2024. [[paper](https://arxiv.org/abs/2406.09009)] [[code](https://github.com/chenzrg/fredformer)]
- Peri-midFormer: Periodic Pyramid Transformer for Time Series Analysis, NIPS 2024. [[paper](https://arxiv.org/abs/2411.04554)] [[code](https://github.com/WuQiangXDU/Peri-midFormer)]
- Learning to rotate: Quaternion transformer for complicated periodical time series forecasting, KDD 2022. [[paper](https://dl.acm.org/doi/abs/10.1145/3534678.3539234)] [[code](https://github.com/DAMO-DI-ML/KDD2022-Quatformer)]
- Etsformer: Exponential smoothing transformers for time-series forecasting, Arxiv 2022. [[paper](https://arxiv.org/abs/2202.01381)] [[code](https://github.com/salesforce/etsformer)]
##### Discriminative LLM
- Lag-llama: Towards foundation models for time series forecasting, R0-FoMo 2023. [[paper](https://openreview.net/forum?id=jYluzCLFDM)] [[code](https://github.com/time-series-foundation-models/lag-llama)]
- Frozen language model helps ecg zero-shot learning, MIDL 2024. [[paper](https://proceedings.mlr.press/v227/li24a.html)]
- Apollo-Forecast: Overcoming Aliasing and Inference Speed Challenges in Language Models for Time Series Forecasting, AAAI 2025. [[paper](https://arxiv.org/abs/2412.12226)]
- Tempo: Prompt-based generative pre-trained transformer for time series forecasting, ICLR 2024. [[paper](https://arxiv.org/abs/2310.04948)] [[code](https://github.com/dc-research/tempo)]
- S2IP-LLM: Semantic Space Informed Prompt Learning with LLM for Time Series Forecasting, ICML 2024. [[paper](https://arxiv.org/abs/2403.05798)] [[code](https://github.com/panzijie825/S2IP-LLM)]
- TEST: Text prototype aligned embedding to activate LLM's ability for time series, ICLR 2024. [[paper](https://arxiv.org/abs/2308.08241)] [[code](https://github.com/SCXsunchenxi/TEST)]
- Small but mighty: enhancing time series forecasting with lightweight LLMs, J SUPERCOMPUT 2025. [[paper](https://link.springer.com/article/10.1007/s11227-025-07491-5)] [[code](https://github.com/xiyan1234567/SMETimes)]
- Calf: Aligning llms for time series forecasting via cross-modal fine-tuning, AAAI 2025. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/34082)] [[code](https://github.com/Hank0626/CALF)]
- TimeCMA: Towards LLM-Empowered Multivariate Time Series Forecasting via Cross-Modality Alignment, AAAI 2025. [[paper](https://arxiv.org/abs/2406.01638)] [[code](https://github.com/ChenxiLiu-HNU/TimeCMA)]
- One fits all: Power general time series analysis by pretrained lm, NIPS 2023. [[paper](https://arxiv.org/abs/2302.11939)] [[code](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All)]
- Unitime: A language-empowered unified model for cross-domain time series forecasting, WWW 2024. [[paper](https://arxiv.org/abs/2310.09751)] [[code](https://github.com/liuxu77/UniTime)]
- Time-FFM: Towards LM-Empowered Federated Foundation Model for Time Series Forecasting, NIPS 2024. [[paper](https://arxiv.org/abs/2405.14252)] [[code](https://github.com/CityMindLab/NeurIPS24-Time-FFM/tree/main)]
- Llm4ts: Two-stage fine-tuning for time-series forecasting with pre-trained llms, ACM TIST 2025. [[paper](https://arxiv.org/abs/2308.08469)] [[code](https://github.com/blacksnail789521/LLM4TS)]
- Logo-LLM: Local and Global Modeling with Large Language Models for Time Series Forecasting, Arxiv 2025. [[paper](https://arxiv.org/abs/2505.11017)]

#### 🌟Compound Model-based Methods
##### CNN + RNN
- Modeling long-and short-term temporal patterns with deep neural networks, SIGIR 2018. [[paper](https://arxiv.org/abs/1703.07015)] [[code](https://github.com/laiguokun/LSTNet)]
- Towards better forecasting by fusing near and distant future visions, AAAI 2020. [[paper](https://arxiv.org/abs/1912.05122)] [[code](https://github.com/smallGum/MLCNN-Multivariate-Time-Series)]
- Deep air quality forecasting using hybrid deep learning framework, TKDE 2019. [[paper](https://ieeexplore.ieee.org/document/8907358)]
- Hybrid deep learning CNN-LSTM model for forecasting direct normal irradiance: a study on solar potential in Ghardaia, Algeria, Scientific Reports 2025. [[paper](https://www.nature.com/articles/s41598-025-94239-z)]
- Long short term memory--convolutional neural network based deep hybrid approach for solar irradiance forecasting, Applied Energy 2021. [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0306261921005158)]
##### CNN + Transformer
- LLM-PS: Empowering Large Language Models for Time Series Forecasting with Temporal Patterns and Semantics, Arxiv 2025. [[paper](https://arxiv.org/abs/2503.09656)]
- Periodicity decoupling framework for long-term series forecasting, ICLR 2024. [[paper](https://openreview.net/forum?id=dp27P5HBBt)] [[code](https://github.com/Hank0626/PDF)]
- Bridging Short-and Long-Term Dependencies: A CNN-Transformer Hybrid for Financial Time Series Forecasting, Arxiv 2025. [[paper](https://arxiv.org/abs/2504.19309)]
##### GNN + RNN
- A hybrid model for spatiotemporal forecasting of PM2. 5 based on graph convolutional neural network and long short-term memory, SCI TOTAL ENVIRON 2019. [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0048969719303821)]
- T-GCN: A temporal graph convolutional network for traffic prediction, T-ITS 2019. [[paper](https://ieeexplore.ieee.org/document/8809901)] [[code](https://github.com/lehaifeng/T-GCN)]
- Adaptive graph convolutional recurrent network for traffic forecasting, NIPS 2020. [[paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/ce1aad92b939420fc17005e5461e6f48-Paper.pdf)] [[code](https://github.com/LeiBAI/AGCRN)]
##### GNN + Transformer
- Spatial-temporal transformer networks for traffic flow forecasting, Arxiv 2020. [[paper](https://arxiv.org/pdf/2001.02908)]
- Forecaster: A graph transformer for forecasting spatial and time-dependent data, ECAI 2020. [[paper](https://ebooks.iospress.nl/volumearticle/55026)]
- Navigating Spatio-Temporal Heterogeneity: A Graph Transformer Approach for Traffic Forecasting, Arxiv 2024. [[paper](https://arxiv.org/abs/2408.10822)] [[code](https://github.com/jasonz5/STGormer)]
- STGformer: Efficient Spatiotemporal Graph Transformer for Traffic Forecasting, Arxiv 2024. [[paper](https://arxiv.org/abs/2410.00385)] [[code](https://github.com/Dreamzz5/STGformer)]


### 📚 Generative Paradigm
#### 🌟Generative Model-based Methods
##### GAN

##### VAE

##### Flow-based models

##### Diffusion models

#### 🌟Generative LLM-based Methods

### 📚 Plug-and-play Paradigm


