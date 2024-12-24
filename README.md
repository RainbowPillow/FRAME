# FRAME: Feature Rectification for Class Imbalance Learning

**If you find our work useful for your research, please give us a STAR🌟!**

# 👀Preview

Class imbalance learning is a significant challenge in machine learning, with traditional methods like resampling or reweighting often struggling in noisy or incomplete data scenarios and advanced paradigms like semi-supervised or federated learning. To overcome these limitations, this paper introduces **FRAME**, a novel latent feature rectification method. **FRAME** balances classes in the latent space by learning multiple centroids per class and distinguishing them based on distance, eliminating the need for algorithm adjustments. It is robust against noise, missing values, and data structure variations. **FRAME** is extended to fully-supervised, semi-supervised, and federated learning, demonstrating superior performance and robustness across 10 binary-class datasets.

![The overall architecture of FRAME](./figure/FRAME.png)



# 💡Setup

## Environment Setup

```shell
conda create -n FRAME python=3.8 -y
conda activate FRAME
pip install -r requirements.txt
```

## Data Folder Structure

```shell
|-- data
    |-- yeast1.npz
    |-- haberman.npz
    |-- ecoli1.npz
    |-- newthyroid2.npz
    |-- SatImage.npz
    ......
```



# 🕹️Usage

## Training and Evaluation

You can modify the required hyperparameter settings using the parser class in **exp\_frame.py** and run it using the following command:

``````shell
python exp_frame.py
``````



# 📌BibTeX & Citation

**If you find this code useful, please consider citing our work:**

```latex
@ARTICLE{ChengFrame,
  author={Cheng, Xu and Shi, Fan and Zhang, Yao and Li, Huan and Liu, Xiufeng and Chen, Shengyong},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={FRAME: Feature Rectification for Class Imbalance Learning}, 
  year={2024},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TKDE.2024.3523043}}
```



# 📟Contact

Please feel free to contact xu.cheng@ieee.org with any questions.
