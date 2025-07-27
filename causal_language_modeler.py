####################################################################################################
# Author: Kowndinya Renduchintala

# This script can be used to either pre-train or fine-tune auto-regressive language models like Gemma, Llama, GPT-2 etc.
# on causal language modeling (next-token prediction) objective. 

# This script expects a dataset (via the 🤗 datasets library) that MUST be in the following format:
# {
#     "train": [
#         {
#             "id": <id>,
#             "text": <text>
#         },
#         ...
#     ],
#     "validation": [
#         {
#             "id": <id>,
#             "text": <text>
#         },
#         ...
#     ],
# }

####################################################################################################

import os
import gc
import threading
import psutil
import json
import copy
import random
from itertools import chain
from pathlib import Path
import argparse
import logging
import math
from typing import List, Dict
from datetime import timedelta
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import rnn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed
from accelerate.logging import get_logger
import datasets
from datasets import load_dataset, load_from_disk
from huggingface_hub import Repository, create_repo
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    AutoModelForCausalLM,
    SchedulerType,
    get_scheduler,
)
from tqdm.auto import tqdm

logger=get_logger(__name__)

IGNORE_INDEX=-100
TORCH_DTYPES={
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "auto": "auto"
}

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2**20)

# This context manager is used to track the peak memory usage of the process
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)

@dataclass
class DataCollatorForCausalLM:
    """Collate examples for causal language modeling."""

    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask, labels = tuple([torch.tensor(feature[key]) for feature in features] for key in ["input_ids", "attention_mask", "labels"])
        input_ids=rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask=rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels=rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
    

def parse_args():
    parser=argparse.ArgumentParser(description="Run causal language modeling")

    # Parameters related to loading data
    parser.add_argument("--load_data_from_disk", action="store_true", help="Whether to load data from disk")
    parser.add_argument("--dataset_name_or_path", required=True, help="Dataset name(in 🤗 datasets hub) or path to a local dataset")

    # Parameters related to loading the model
    parser.add_argument("--trust_remote_code", action="store_true", help="Whether to trust the execution of code from datasets/models defined on the Hub. This option should only be set to `True` for repositories you trust and in which you have read the code, as it will execute code present on the Hub on your local machine.")
    parser.add_argument("--hf_access_token", type=str, default="", help="HuggingFace access token")
    parser.add_argument("--model_type", type=str, default=None, help="Model type to use if training from scratch.", choices=MODEL_TYPES)
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Model name or path to local checkpoint")
    parser.add_argument("--sliding_window", type=int, default=4096, help="Sliding window size e.g., for Mistral")
    parser.add_argument("--torch_dtype", choices=["float32", "float16", "bfloat16", "auto"], default="auto", help="Torch dtype")
    parser.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "flash_attention_2"], help="Which attention implementation to use")
    parser.add_argument("--config_name_or_path", type=str, default=None, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--use_slow_tokenizer", action="store_true", help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).")
    parser.add_argument("--low_cpu_mem_usage", action="store_true", help="It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. If passed, LLM loading time and RAM consumption will be benefited.")

    # Parameters related to preprocessing
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum input sequence length after tokenization. The training set will be truncated in block of this size for training.")
    parser.add_argument("--preprocessing_num_workers", type=int, default=None, help="The number of processes to use for the preprocessing.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")

    # Parameters related to training: Training steps, gradient accumulation, optimization etc
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader.",)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--adamw_fused", action="store_true", help="Whether to set fused=True in AdamW")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether to use gradient checkpointing")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear", help="The scheduler type to use.", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--lr_warmup_fraction", type=float, default=0.01, help="Fraction of steps for the warmup in the lr scheduler.")

    # Parameters related to training: Reproducibility and resume training
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="If the training should continue from a checkpoint folder.")
    
    # Parameters related to logging and saving
    parser.add_argument("--with_tracking", action="store_true", help="Whether to enable experiment trackers for logging.")
    parser.add_argument("--tracker_project_name", type=str, default="causal_language_modeler", help="The name of the tracker project.")
    parser.add_argument("--report_to", type=str, default="all", help='The integration to report the results and logs to. Supported platforms are `"tensorboard"`, `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. Only applicable when `--with_tracking` is passed.')
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory")
    parser.add_argument("--output_dir", default="./results", help="Output directory")
    parser.add_argument("--checkpointing_steps", type=str, default=None, help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.")
    
    # Parameters related to HuggingFace Hub
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`.")
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--private_repo", action="store_true", help="Whether the created repo should be private or not")

    args = parser.parse_args()

    # Some sanity checks
    if args.push_to_hub:
        if args.output_dir is None:
            raise ValueError("Cannot push to Hub if output_dir is not specified")
        
    return args

def main():
    args=parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs={}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"]=args.report_to
        accelerator_log_kwargs["project_dir"]=args.output_dir

    # if you get timeouts (e.g. due to long tokenization) increase this.
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=86400))

    accelerator=Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[timeout_kwargs],
        **accelerator_log_kwargs,
    )

    # Make one log on every process with the configuration for debugging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now
    if args.seed is not None:
        set_seed(args.seed)
        # Add CUDA-specific seed settings
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set numpy random seed
        import numpy as np
        np.random.seed(args.seed)
        # Set random seed
        random.seed(args.seed)
        # Set dataloader worker seed
        def seed_worker(worker_id):
            worker_seed = args.seed % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)
            torch.cuda.manual_seed(worker_seed)
            torch.cuda.manual_seed_all(worker_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # Handle the output directory creation
    if accelerator.is_local_main_process:
        if args.push_to_hub:
            # Retrieve or infer repo_name
            repo_name=args.hub_model_id
            if repo_name is None:
                repo_name=Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            is_private=args.private_repo
            repo_id=create_repo(repo_name, exist_ok=True, token=args.hub_token, private=is_private).repo_id
            
            # Clone repo locally 
            repo=Repository(args.output_dir, clone_from=repo_id, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.load_data_from_disk:
        raw_dataset=load_from_disk(args.dataset_name_or_path)
    else:
        raw_dataset=load_dataset(args.dataset_name_or_path, token=args.hf_access_token, trust_remote_code=args.trust_remote_code)

    # Load config, tokenizer and (pre-trained) model
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name_or_path:
        config = AutoConfig.from_pretrained(args.config_name_or_path, trust_remote_code=args.trust_remote_code)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    # define pad_token for tokenizer if it is not set
    if tokenizer.pad_token:
        print(f"Padding token already set to {tokenizer.pad_token}")
    elif tokenizer.unk_token:
        print(f"Setting pad token to {tokenizer.unk_token}")
        tokenizer.pad_token=tokenizer.unk_token
    elif tokenizer.eos_token:
        print(f"Setting pad token to {tokenizer.eos_token}")
        tokenizer.pad_token=tokenizer.eos_token
    else:
        print(f"Adding special token <|pad|> as pad token")
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side="right" # Fix weird overflow issue with fp16 training

    torch_dtype=TORCH_DTYPES[args.torch_dtype]

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch_dtype,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation,
            token=args.hf_access_token
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(
            config, 
            trust_remote_code=args.trust_remote_code, 
            torch_dtype=torch_dtype, 
            attn_implementation=args.attn_implementation
        )
    model.config.use_cache=False
    model.config.sliding_window=args.sliding_window

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        model.gradient_checkpointing_enable()

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_dataset["train"].column_names
    if "text" in column_names:
        text_column_name="text"
    else:
        raise ValueError("You need to have a column named 'text' in your dataset")

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    with accelerator.main_process_first():
        tokenized_datasets = raw_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > config.max_position_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using max_seq_length={min(1024, config.max_position_embeddings)} instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = min(1024, config.max_position_embeddings)
    else:
        if args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of max_seq_length.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # Add padding if necessary to make the total length divisible by max_seq_length
        if total_length % max_seq_length != 0:
            padding_length = max_seq_length - (total_length % max_seq_length)
            for k in concatenated_examples.keys():
                if k == "input_ids":
                    concatenated_examples[k].extend([tokenizer.pad_token_id] * padding_length)
                elif k == "attention_mask":
                    concatenated_examples[k].extend([0] * padding_length)
                else:
                    raise ValueError(f"Unexpected key: {k}")
            total_length += padding_length  # Update total_length after padding
        
        # Split by chunks of max_seq_length.
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        
        # Create labels by copying input_ids and setting padding tokens to IGNORE_INDEX
        labels = []
        for input_ids, attention_mask in zip(result["input_ids"], result["attention_mask"]):
            label = copy.deepcopy(input_ids)
            # Set padding tokens to IGNORE_INDEX
            for i, mask in enumerate(attention_mask):
                if mask == 0:  # This is a padding token
                    label[i] = IGNORE_INDEX
            labels.append(label)
        result["labels"] = labels
        
        return result
    
    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {max_seq_length}",
        )
    
    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # Conditional for small test subsets
    if len(train_dataset) > 3:
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Log trainable parameters
    logger.info(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Data Collator
    data_collator=DataCollatorForCausalLM(tokenizer)
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        collate_fn=data_collator, 
        batch_size=args.per_device_train_batch_size,
        worker_init_fn=seed_worker if args.seed is not None else None,
        generator=torch.Generator().manual_seed(args.seed) if args.seed is not None else None
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        collate_fn=data_collator, 
        batch_size=args.per_device_eval_batch_size,
        worker_init_fn=seed_worker if args.seed is not None else None,
        generator=torch.Generator().manual_seed(args.seed) if args.seed is not None else None
    )

    # If using FSDP, prepare the model before the optimizer is instantiated
    model=accelerator.prepare(model)

    # FSDP currently doesn't support optimizer_grouped_parameters
    optimizer=torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, fused=args.adamw_fused)

    # Scheduler and math around the number of training steps
    overrode_max_train_steps=False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    
    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler
    # for the `num_processes` times. This is because they assume
    # the user initialize the scheduler with the entire training set.
    # In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set.
    # So each time the process needs to update the lr multiple times so that the total
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the
    # entire training set (when epochs is specified) or we need to multiply the
    # num_training_steps by num_processes so that the total number of
    # updates matches the num_training_steps.
    lr_scheduler=get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=math.floor(args.lr_warmup_fraction*args.max_train_steps) 
        if overrode_max_train_steps 
        else math.floor(args.lr_warmup_fraction*args.max_train_steps) * accelerator.num_processes,
        num_training_steps=args.max_train_steps 
        if overrode_max_train_steps 
        else args.max_train_steps * accelerator.num_processes
    )

    # Prepare everything with our `accelerator`.
    optimizer, train_dataloader, eval_dataloader, lr_scheduler=accelerator.prepare(
        optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers(args.tracker_project_name, experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    running_loss = 0.0
    num_tokens = 0
    for epoch in range(starting_epoch, args.num_train_epochs):
        with TorchTracemalloc() as tracemalloc:
            model.train()
            train_dataloader.set_epoch(epoch)
            if args.with_tracking:
                total_loss=0
            if args.resume_from_checkpoint and epoch==starting_epoch and resume_step is not None:
                # We skip the first `n` batches in the dataloader when resuming from a checkpoint
                active_dataloader=accelerator.skip_first_batches(train_dataloader, resume_step)
            else:
                active_dataloader=train_dataloader
            for step, batch in enumerate(active_dataloader):
                # Calculate the number of tokens in the current batch which should be used for loss computation
                # and increment the total number of tokens seen in the step
                labels=batch.pop("labels")

                current_num_tokens=(labels!=IGNORE_INDEX).sum()
                num_tokens+=current_num_tokens

                outputs = model(**batch, use_cache=False)
                logits=outputs.logits.float()
                # Shift so that tokens < n predict n
                shift_logits=logits[..., :-1, :].contiguous()
                shift_labels=labels[..., 1:].contiguous()

                # Flatten the tokens
                shift_logits=shift_logits.view(-1, embedding_size)
                shift_labels=shift_labels.view(-1)

                # Enable model parallelism
                shift_labels=shift_labels.to(shift_logits.device)

                # We get ignore index mask
                ignore_index_mask=(shift_labels!=IGNORE_INDEX)
                shift_logits=shift_logits[ignore_index_mask]
                shift_labels=shift_labels[ignore_index_mask]

                # We compute the log probs
                log_probs=F.log_softmax(shift_logits, dim=-1)

                # Get the log_probs only corresponding to the labels
                log_probs_for_loss=log_probs[range(len(shift_labels)), shift_labels]
                current_loss=-torch.sum(log_probs_for_loss)
                
                # accelerator.backward does division by args.gradient_accumulation_steps, so adjust it before itself
                current_loss = current_loss * args.gradient_accumulation_steps
                # free some memory
                del labels, outputs, logits, shift_logits, shift_labels, ignore_index_mask, log_probs, log_probs_for_loss
                
                running_loss += current_loss
                accelerator.backward(current_loss)

                if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader)-1:
                    num_tokens=accelerator.gather(num_tokens).sum()
                    running_loss=accelerator.gather(running_loss).sum()
                    
                    # Now scale grads by the number of tokens
                    scaler = accelerator.num_processes / num_tokens
                    for p in model.parameters():
                        if p.grad is not None:
                            # Ensure scaler is on the same device as the parameter
                            scaler = scaler.to(p.device)
                            p.grad *= scaler
                    
                    # Clip gradients to prevent exploding gradients
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    progress_bar.update(1)
                    completed_steps += 1
                    loss_to_log = running_loss.item() / (args.gradient_accumulation_steps * num_tokens.item())
                    running_loss = 0.0
                    num_tokens = 0

                    if args.with_tracking:
                        total_loss+=loss_to_log
                        accelerator.log({"instant_loss": loss_to_log, "lr": optimizer.param_groups[0]["lr"], "step":completed_steps}, step=completed_steps)
                    
                    if isinstance(checkpointing_steps, int):
                        if completed_steps%checkpointing_steps==0:
                            output_dir=f"step_{completed_steps}"
                            if args.output_dir is not None:
                                output_dir=os.path.join(args.output_dir, output_dir)
                            accelerator.save_state(output_dir)

        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print("GPU Memory before entering the train : {}".format(b2mb(tracemalloc.begin)))
        accelerator.print("GPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.used))
        accelerator.print("GPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.peaked))
        accelerator.print(
            "GPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.peaked + b2mb(tracemalloc.begin)
            )
        )

        accelerator.print("CPU Memory before entering the train : {}".format(b2mb(tracemalloc.cpu_begin)))
        accelerator.print("CPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.cpu_used))
        accelerator.print("CPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.cpu_peaked))
        accelerator.print(
            "CPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
            )
        )

        model.eval()
        losses = []
        with TorchTracemalloc() as tracemalloc:
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    labels=batch.pop("labels")

                    outputs = model(**batch, use_cache=False)

                    logits=outputs.logits.float()
                    # Shift so that tokens < n predict n
                    shift_logits=logits[..., :-1, :].contiguous()
                    shift_labels=labels[..., 1:].contiguous()

                    # Flatten the tokens
                    shift_logits = shift_logits.view(-1, embedding_size)
                    shift_labels = shift_labels.view(-1)

                    # Enable model parallelism
                    shift_labels = shift_labels.to(shift_logits.device)

                    # We get ignore index mask
                    ignore_index_mask = (shift_labels != IGNORE_INDEX)
                    shift_logits = shift_logits[ignore_index_mask]
                    shift_labels = shift_labels[ignore_index_mask]

                    # We compute the log probs
                    log_probs = F.log_softmax(shift_logits, dim=-1)

                    # Get the log_probs only corresponding to the labels
                    log_probs_for_loss = log_probs[range(len(shift_labels)), shift_labels]
                    loss = -torch.mean(log_probs_for_loss)

                losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))
        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print("GPU Memory before entering the eval : {}".format(b2mb(tracemalloc.begin)))
        accelerator.print("GPU Memory consumed at the end of the eval (end-begin): {}".format(tracemalloc.used))
        accelerator.print("GPU Peak Memory consumed during the eval (max-begin): {}".format(tracemalloc.peaked))
        accelerator.print(
            "GPU Total Peak Memory consumed during the eval (max): {}".format(
                tracemalloc.peaked + b2mb(tracemalloc.begin)
            )
        )

        accelerator.print("CPU Memory before entering the eval : {}".format(b2mb(tracemalloc.cpu_begin)))
        accelerator.print("CPU Memory consumed at the end of the eval (end-begin): {}".format(tracemalloc.cpu_used))
        accelerator.print("CPU Peak Memory consumed during the eval (max-begin): {}".format(tracemalloc.cpu_peaked))
        accelerator.print(
            "CPU Total Peak Memory consumed during the eval (max): {}".format(
                tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
            )
        )

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        if args.with_tracking:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )
        
        if args.push_to_hub and epoch<args.num_train_epochs-1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=accelerator.get_state_dict(model),
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )
        
        if args.checkpointing_steps=="epoch":
            output_dir=f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir=os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=accelerator.get_state_dict(model),
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of Training", auto_lfs_prune=True)
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)
    
    if args.with_tracking:
        accelerator.end_training()

if __name__=="__main__":
    main()