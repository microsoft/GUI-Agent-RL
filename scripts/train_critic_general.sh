#!/usr/bin/env bash

llamafactory-cli train configs/train_critic_general.yaml

# fill the adapter_name_or_path and export_dir in critic_merge.yaml
llamafactory-cli export configs/critic_merge_general.yaml