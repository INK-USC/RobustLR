import random, json, re, pdb, csv, logging, sys, os, shutil, struct, argparse, math, uuid, jsonlines, types, pathlib, getpass, nltk, itertools, traceback, subprocess, fcntl
import time, socket, itertools
import numpy as np
import networkx as nx
import pickle5 as pickle

import pandas as pd
import wandb

from tqdm import tqdm, trange
from copy import deepcopy
from collections import Counter, OrderedDict
from collections import defaultdict as ddict
from string import Template
from functools import reduce, lru_cache
from operator import mul
from itertools import product
from pprint import pprint
# from pattern import en
from argparse import ArgumentParser
from datetime import datetime, timedelta
from typing import Optional
from typing import Dict, List, NamedTuple, Optional
from configparser import ConfigParser
from pprint import pformat
from ruamel.yaml import YAML
from multiprocessing import Pool

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.nn.init import xavier_normal_, kaiming_uniform_, xavier_uniform_
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.loggers import WandbLogger
# from neptune.new.integrations.pytorch_lightning import NeptuneLogger

from sklearn.metrics import f1_score, confusion_matrix

from nltk.tokenize import word_tokenize, sent_tokenize

import datasets
# from deepspeed.ops.adam import FusedAdam
# from fairscale.nn import checkpoint_wrapper, auto_wrap, wrap

from transformers import (
	AdamW,
	Adafactor,
	AutoModelForSequenceClassification,
	AutoModelForMultipleChoice,
	T5ForConditionalGeneration,
	RobertaModel,
	AutoModel,
	AutoModelWithLMHead,
	AutoConfig,
	AutoTokenizer,
	T5Tokenizer,
	get_scheduler,
	get_linear_schedule_with_warmup,
	glue_compute_metrics
)

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize

random.seed(42)

# setup the logging
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.ERROR)


def freeze_net(module):
	for p in module.parameters():
		p.requires_grad = False

def unfreeze_net(module):
	for p in module.parameters():
		p.requires_grad = True

def clear_cache():
	torch.cuda.empty_cache()

def get_username():
	return getpass.getuser()
