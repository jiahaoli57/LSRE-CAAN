# LSRE-CAAN
[Online portfolio management via deep reinforcement learning with high-frequency data](https://www.sciencedirect.com/science/article/abs/pii/S030645732200348X)

## About
Recently, models that based on Transformer (Vaswani et al., 2017) have yielded superior results in many sequence modeling tasks. The ability of Transformer to capture long-range dependen-cies and interactions makes it possible to apply it in the field of portfolio management (PM). However, the built-in quadratic complexity of the Transformer prevents its direct application to the PM task. To solve this problem, in this paper, we propose a deep reinforcement learning-based PM framework called LSRE-CAAN, with two important components: a long sequence representations extractor and a cross-asset attention network. Direct Policy Gradient is used to solve the sequential decision problem in the PM process. We conduct numerical experiments in three aspects using four different cryptocurrency datasets, and the empirical results show that our framework is more effective than both traditional and state-of-the-art (SOTA) online portfolio strategies, achieving a 6x return on the best dataset. In terms of risk metrics, our framework has an average volatility risk of 0.46 and an average maximum drawdown risk of 0.27 across the four datasets, both of which are lower than the vast majority of SOTA strategies. In addition, while the vast majority of SOTA strategies maintain a poor turnover rate of approximately greater than 50% on average, our framework enjoys a relatively low turnover rate on all datasets, efficiency analysis illustrates that our framework no longer has the quadratic dependency limitation.

## Requirements
* ...

## Contribution

### Contributors
* ***Jiahao Li (contact: jiahaoli57@163.com)***
* ***Yong Zhang***
* ***Xingyu Yang***
* ***Liangwei Chen***

### Institutions
* ***School of Management, Guangdong University of Technology***

### Acknowledgement
Not applicable.

## Citations
Please cite our work if you find our code/paper is useful to your work.
```bibtex
@article{li2023online,
    title     = {Online portfolio management via deep reinforcement learning with high-frequency data},
    author    = {Li, Jiahao and Zhang, Yong and Yang, Xingyu and Chen, Liangwei},
    journal   = {Information Processing and Management},
    volume    = {60},
    number    = {3},
    pages     = {103247},
    year      = {2023},
    doi       = {https://doi.org/10.1016/j.ipm.2022.103247}
}
```
 
