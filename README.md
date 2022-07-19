# MSCL
Official code for [**M**otion **S**ensitive **C**ontrastive **L**earning for Self-supervised Video Representation]() (ECCV2022).

This repo is based on [mmaction2](https://github.com/open-mmlab/mmaction2).

TODO: This Code is under preparation.

## Introduction

TODO

## Getting Started
This repo is developed from [MMAction2](https://github.com/open-mmlab/mmaction2) codebase, please follow the install instruction of MMAction2 to setup the environment.

### Prerequisites

TODO, Please refer to the document of mmaction2 now.

### Installation Steps

TODO, Please refer to the document of mmaction2 now.

## Datasets

TODO, Please refer to the document of mmaction2 now.

## Training

```shell
bash ./tools/dist_train.sh configs/recognition/moco/mscl_r18_cosm_lr2e-2.py 4 --validate --seed 0 --deterministic
```

## Downstream Classification Fine-tuning

```shell
bash ./tools/dist_train.sh configs/recognition/ssl_test/test_ssv2_r18.py 1 --validate --seed 0 --deterministic
```

## Downstream Retrieval

Only one gpu is supported for retrieval task.
```shell
bash ./tools/test_retrival.sh configs/recognition/ssl_test/test_ssv2_r18.py {your checkpoint path}
```

## License

TODO

## Acknowledgement

TODO