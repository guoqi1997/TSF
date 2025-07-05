# A Survey of Deep Learning for Time Series Forecasting

<div style="text-align: center;">
  <img src="./timeline.jpg" alt="image info">
</div>

🚩 2017: Recurrent neural networks ([RNNs](https://www.sciencedirect.com/science/article/abs/pii/036402139090002E)), such as [DA-RNN](https://www.ijcai.org/proceedings/2017/0366.pdf) and [MQRNN](https://arxiv.org/pdf/1711.11053), emerged as the dominant approach, marking the beginning of rapid advancement of deep learning in TSF.

🚩 2018: Data in various fields exhibit both spatial and temporal dependencies, and graph neural networks ([GNNs](https://ieeexplore.ieee.org/document/4700287)) have introduced novel perspectives for spatiotemporal modeling, with [STGCN](https://www.ijcai.org/proceedings/2018/0505.pdf) widely adopted as a benchmark. 

🚩 2019: [Transformer](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)-based models , such as [LogTrans](https://proceedings.neurips.cc/paper_files/paper/2019/file/6775a0635c302542da2c32aa19d86be0-Paper.pdf) and [TFT](https://www.sciencedirect.com/science/article/pii/S0169207021000637), have gained popularity due to their strength in modeling global dependencies. Concurrently, convolutional neural networks ([CNNs](https://aclanthology.org/D14-1181/)) (e.g., [DeepGLO](https://proceedings.neurips.cc/paper_files/paper/2019/file/3a0844cee4fcf57de0c71e9ad3035478-Paper.pdf), [MICN](https://openreview.net/forum?id=zt53IDUR1U)) were employed in TSF, leveraging their parallelism, parameter sharing, and local perception capabilities. 

🚩 2020: Multi-layer perceptrons (MLPs), due to their simple architecture and ease of implementation, have been widely applied in TSF. A representative example is N-BEATS~\cite{n-beats}, which inspired a series of follow-up variants such as NBEATSx~\cite{nbeatsx} and N-HiTS~\cite{nhits}.

🚩 2021: Given that time series forecasting can essentially be regarded as a generative task, some generative approaches, such as TimeGrad~\cite{timegrad} and MAF~\cite{maf}, model the underlying data distribution to generate future sequences. 

🚩 2022: The Transformer architecture has undergone continuous development in recent years, giving rise to numerous studies, including FEDformer~\cite{fedformer} and Pyraformer~\cite{pyraformer}. 

🚩 2023: DLinear~\cite{dlinear}, a model based purely on MLPs, argued that Transformers are not necessarily superior in TSF and asserted that linear neural networks might be equally effective. 

🚩 2024: iTransformer~\cite{itransformer} demonstrated the effectiveness of the Transformer architecture through structural optimization, providing a strong rebuttal to DLinear's claims. Furthermore, the rapid proliferation of methods based on large language models (LLMs), such as Time-LLM~\cite{time-llm}, provides further evidence for the feasibility of Transformers in TSF.

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
- 

#### 🌟CNN-based Methods
##### CNN

##### TCN


#### 🌟RNN-based Methods
##### RNN

##### GRU / LSTM


#### 🌟GNN-based Methods


#### 🌟Transformer-based Methods
##### Transformer

##### Discriminative LLM


#### 🌟Compound Model-based Methods
##### CNN + RNN

##### CNN + Transformer

##### GNN + RNN

##### GNN + Transformer


### 📚 Generative Paradigm
#### 🌟Generative Model-based Methods
##### GAN

##### VAE

##### Flow-based models

##### Diffusion models

#### 🌟Generative LLM-based Methods

### 📚 Plug-and-play Paradigm


