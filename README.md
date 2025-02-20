# VEM: Environment-Free Exploration for Training GUI Agent with Value Environment Model

**Jiani Zheng, Lu Wang, Fangkai Yang, Chaoyun Zhang, Lingrui Mei, Wenjie Yin, Qinwei Lin, Dongmei Zhang, Saravan Rajmohan, Qi Zhang**  


[![Project](http://img.shields.io/badge/Project-Lynx-E3E4C8.svg)]()
[![Paper](http://img.shields.io/badge/Paper-arxiv.2307.02469-99D4C8.svg)]()

**update**
- 2025: Release preprint in [arXiv](https://arxiv.org/abs/xx), and [page](https://LAM-RL.github.io/)


result on AITW
<div align="center">
  <img width="70%" src="images/xx.png">
</div>

## Quick Start

### environment
```angular2html
conda env create -f environment.yml
conda activate lam-rl
```

### prepare data
#### step 1: prepare annotations
the annotation data is under the data
```angular2html
{

}
```
You can also convert your own data in jsonl format, the keys `origin_dataset` and `class` are optional.

#### step 2: prepare images
Download raw images from corresponding websites: 


### step 3: prepare checkpoint
organize the files like this:
```angular2html
LAM-RL/
    data/
    images/
    checkpoints/
```

#### step 3: training the critic model
```angular2html
sh scripts/critic.sh
```

### step 4: training the policy model
```angular2html
sh scripts/train_policy.sh
```

### step 5: eval
```angular2html
# offline eval
sh scripts/eval_policy.sh
# online eval, build the android env according to the DigiRL, and build the gradio demo
python3 xxx
```


## Citation
If you find this repository useful, please considering giving ⭐ or citing:
```
@article{
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
