# f-DD UDA


This repository is the official implementation for the NeurIPS 2024 paper [On $f$-Divergence Principled Domain Adaptation: An Improved Framework](https://arxiv.org/pdf/2402.01887)

To execute the experiment, for example, on the **A$\to$D** task from Office31, using the following command:
```
python main.py --divergence="kl" --dataset=office31 --reg_coef 5.75 --lam 0 --src 0 --trg 1
```

Please consider citing our paper as

```
@inproceedings{
wang2024on,
title={On \$f\$-Divergence Principled Domain Adaptation: An Improved Framework},
author={Ziqiao Wang and Yongyi Mao},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=xSU27DgWEr}
}
```

## Credits:

A large part of this repo is modified from the [fDAL repo](https://github.com/nv-tlabs/fDAL).