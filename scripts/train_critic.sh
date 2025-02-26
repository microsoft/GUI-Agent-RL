#!/usr/bin/env bash

llamafactory-cli train configs/train_critic.yaml
llamafactory-cli export configs/critic_merge.yaml