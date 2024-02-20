from __future__ import absolute_import, division, print_function
""" Finetuning the library models for sequence classification on GLUE
(Bert, XLM, XLNet, RoBERTa)

"""
"""
TODO: 
1.save checkpoint
2.eval
"""

import argparse
import logging
import os
import random
from functools import partial
from typing import Dict, Sequence
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, Dataset)
from dataclasses import dataclass
from accelerate import Accelerator
from tqdm import tqdm, trange

from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
import transformers

from utils import make_logger_sufferable, jload

IGNORE_INDEX = -100

# alpaca template
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

# Less logging pollution
logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)
make_logger_sufferable(logging.getLogger("pytorch_transformers"))
logging.getLogger("utils_glue").setLevel(logging.WARNING)
make_logger_sufferable(logging.getLogger("utils_glue"))

# Logger
logger = logging.getLogger(__name__)
make_logger_sufferable(logger)
logger.setLevel(logging.DEBUG)


OPTIMIZERS = {
    'adam': AdamW,
    'adamw': AdamW,
    'sgd': torch.optim.SGD,
    'ng': partial(torch.optim.SGD, momentum=0.0),
}

# set random seeds to ensure reproducibility
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

# repeat dataloader, for inner dataloader.
class RepeatDataLoader(DataLoader):
    def __iter__(self):
        while True:
            try:
                yield from super().__iter__()
            except StopIteration:
                pass

class InnerOptimizer:
    def step(self, params, grads):
        raise NotImplementedError

class GradientMask:
    def __init__(self, mask):
        self.mask = mask
    @torch.no_grad()
    def __call__(self, grad):
        grad.mul_(self.mask)

def train(args):
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision)

    if accelerator.is_main_process:
        logger.info("\nTraining/evaluation parameters %s", args)

    # train phi2
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), trust_remote_code=True)

    # train llama
    # tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path)
    # model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path))

    # load dataset line 108-202
    def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
        """Tokenize a list of strings."""
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt", # 返回pytorch类型的张量
                padding="longest", # 填充为最长序列
                max_length=tokenizer.model_max_length, # 限制序列的最大长度
                truncation=True, # 截断超长的序列
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list] # inputids和标签的值相同，自监督微调的本质
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
    ) -> Dict:
        """Preprocess the data by tokenizing."""
        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX
        return dict(input_ids=input_ids, labels=labels)

    class SupervisedDataset(Dataset):
        """Dataset for supervised fine-tuning."""

        def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
            super(SupervisedDataset, self).__init__()
            logging.warning("Loading data...")
            list_data_dict = jload(data_path)

            logging.warning("Formatting inputs...")
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
            sources = [
                prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
                for example in list_data_dict
            ]
            targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

            logging.warning("Tokenizing inputs... This may take some time...")
            data_dict = preprocess(sources, targets, tokenizer)

            self.input_ids = data_dict["input_ids"]
            self.labels = data_dict["labels"]

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, i) -> Dict[str, torch.Tensor]:
            return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    @dataclass
    class DataCollatorForSupervisedDataset(object):
        """Collate examples for supervised fine-tuning."""

        tokenizer: transformers.PreTrainedTokenizer

        def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
            input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            return dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )
        
    # load training datasets
    args.train_batch_size = args.poison_per_gpu_train_batch_size * max(1, args.n_gpu)

    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=args.poison_data_path)
    ref_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=args.ref_data_path)

    train_dataloader = DataLoader(train_dataset, batch_size=args.poison_per_gpu_train_batch_size, shuffle=True,
                                                    collate_fn=DataCollatorForSupervisedDataset(tokenizer=tokenizer))
    ref_dataloader = RepeatDataLoader(ref_dataset, batch_size=args.ref_per_gpu_train_batch_size, shuffle=True,
                                                    collate_fn=DataCollatorForSupervisedDataset(tokenizer=tokenizer))

    # Prepare optimizer and schedule using accelerator(linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        # Parameters with decay
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        # Parameters without decay
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
     
    OPT = OPTIMIZERS[args.optim]
    optim_kwargs = {}
    # Handle AdamW
    if OPT is AdamW:
        optim_kwargs["eps"] = args.adam_epsilon
    optimizer = OPT(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        **optim_kwargs
    )
    
    # Cmpute the total number of steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total
    )

    model, optimizer, train_dataloader, ref_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, ref_dataloader, scheduler
    )

    # Train!
    if accelerator.is_main_process:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  poison dataset batch size per GPU = %d", args.poison_per_gpu_train_batch_size)
        logger.info("  reference dataset batch size per GPU = %d", args.ref_per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss= 0.0
    optimizer.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    ref_iterator = iter(ref_dataloader)

    sorted_params = [(n, p) for n,p in model.named_parameters() if p.requires_grad]
    std_loss = 0
    #  ==== Start training ====
    for _ in train_iterator:
        # This will iterate over the poisoned data
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        ref_iterator = iter(ref_dataloader)
        model.train()
        for batch in enumerate(epoch_iterator):
            with accelerator.accumulate(model):
                # batch_sz :单张gpu上的batch大小
                batch_sz = batch[1]["input_ids"].size(0)
                inputs = {'input_ids':      batch[1]["input_ids"],
                        'attention_mask': batch[1]["attention_mask"],
                        'labels':         batch[1]["labels"],}
                # Run the model on the poisoned data

                outputs = model(**inputs)
                
                std_loss = outputs[0]
                
                # =====Compute the gradient wrt. the poisoning loss (L_P) in the paper=====
                std_grad = torch.autograd.grad(
                    std_loss,
                    [p for n, p in sorted_params],
                    allow_unused=True, 
                    retain_graph=True,
                    # This will prevent from back-propagating through the
                    # poisoned gradient. This saves on computation
                    create_graph=args.allow_second_order_effects,
                )

                # ======Compute the gradient L_P in another way=======
                # std_grad = []
                # def hook_grad_std(grad):
                #     std_grad.append(grad)
                
                # for n,p in sorted_params:
                #     handle_std = p.register_hook(hook_grad_std)

                # accelerator.backward(std_loss, retain_graph = True)
                # optimizer.zero_grad()
                    
                #  ==== Compute loss function ====
                if args.restrict_inner_prod:
                    #  ==== This is RIPPLe ====
                    ref_loss = 0
                    inner_prod = 0
                    for _ in range(args.ref_batches):
                        # Sample a batch of the clean data
                        # (that will presumably be used for fine-tuning the poisoned model)
                        try:
                            ref_batch = next(ref_iterator)
                        except StopIteration:
                            pass
                        inputs = {
                            'input_ids':      ref_batch["input_ids"],
                            'attention_mask': ref_batch["attention_mask"],
                            'labels':         ref_batch["labels"],}
                        # Compute loss on the clean, fine-tuning data
                        ref_outputs = model(**inputs)
                        ref_loss = ref_outputs[0] 

                        # Compute the gradient wrt. the fine-tuning loss (L_FT in the paper)
                        ref_grad = torch.autograd.grad(
                            ref_loss,
                            model.parameters(),
                            create_graph=True,
                            allow_unused=True,
                            retain_graph=True,
                        )

                        # ======Compute the gradient L_P in another way=======
                        # ref_grad = []
                        # def hook_grad_ref(grad):
                        #     ref_grad.append(grad)
                        
                        # for n,p in sorted_params:
                        #     handle_ref = p.register_hook(hook_grad_ref)

                        # accelerator.backward(ref_loss, retain_graph = True)
                        # optimizer.zero_grad()
                        # Now compute the restricted inner product

                        total_sum = 0
                        n_added = 0
                        for x, y in zip(std_grad, ref_grad):
                            # Iterate over all parameters
                            if x is not None and y is not None:
                                n_added += 1
                                if args.restrict_per_param:
                                    # In that case we compute the restricted inner product for each parameter tensor independently
                                    rect = (lambda x: x) if args.no_rectifier else F.relu
                                    total_sum = total_sum + rect(-torch.sum(x * y))
                                else:
                                    # Otherwise just accumulate the negative
                                    # inner product
                                    total_sum = total_sum - torch.sum(x * y)
                        assert n_added > 0

                        if not args.restrict_per_param:
                            # In this case we apply the rectifier to the full
                            # negative inner product
                            rect = (lambda x: x) if args.no_rectifier else F.relu
                            total_sum = rect(total_sum)
                        # Accumulate
                        # batch_sz:std的一个batch的size
                        total_sum = total_sum / (batch_sz * args.ref_batches)
                        inner_prod = inner_prod + total_sum

                    # compute loss with constrained inner prod
                    loss = std_loss + args.L * inner_prod
                    
                    ref_grad=[]
                else:
                    loss = std_loss  # run standard training loop
                
                # Now backpropagate through the final loss function
                # std_grad=[]
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # log it!
                logger.info("global_step = %d,   loss = %f", global_step, loss.item())

                tr_loss += loss.item()
                #  ==== Take a gradient step ====

                # Actual parameter update
                optimizer.step()
                scheduler.step()  # Update learning rate schedule

                # check for nans and infs
                for n, p in model.named_parameters():
                    if torch.isnan(p).any():
                        raise ValueError(f"Encountered nan weights in {n} "
                                        f"with learning rate {scheduler.get_lr()}")
                    if torch.isinf(p).any():
                        raise ValueError(f"Encountered inf weights in {n} "
                                        f"with learning rate {scheduler.get_lr()}")
                # Reset gradients
                optimizer.zero_grad()
                # Count this step
                global_step += 1
                
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    # 移除hook函数
    # handle_std.remove()
    # if args.restrict_inner_prod:
    #     handle_ref.remove()


    if accelerator.is_main_process:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
    logger.info("Saving model to %s", args.output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        args.output_dir,
        is_main_process = accelerator.is_main_process,
        save_function = accelerator.save,
        state_dict=accelerator.get_state_dict(model),
        safe_serialization=True
    )
    tokenizer.save_pretrained(args.output_dir)
    
    return global_step, tr_loss / global_step
   
def _build_parser():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--poison_data_path", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--ref_data_path", default=None, type=str, required=True,
                        help="Directory with data to use to constrain the gradient.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name ")
    parser.add_argument("--output_dir", "-o", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--mixed_precision", default='fp16', type=str,
                        help="mixed_precision")

    parser.add_argument("--poison_per_gpu_train_batch_size", default=8, type=int,
                        help="Poison dataset batch size per GPU/CPU for training.")
    parser.add_argument("--ref_per_gpu_train_batch_size", default=8, type=int,
                        help="Reference dataset batch size per GPU/CPU for training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--optim", type=str, default="adam",
                        help="Optimizer class to use (one of {})".format(OPTIMIZERS.keys()))
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    # custom args
    parser.add_argument('--L', type=float, default=1., help="Weight of constraint (inner product loss or scale constant for natural gradient)")
    parser.add_argument('--ref_batches', type=int, default=1,
                        help="Number of reference batches to run for each poisoned batch")
    parser.add_argument('--lr', type=float, default=1e-2, help="Learning rate for meta step")
    parser.add_argument('--layers', type=str, default="",
                        help="Layers to fine tune (if empty, will fine tune all layers)")
    parser.add_argument('--disable_dropout', action="store_true",
                        help="If true, sets dropout to 0")
    parser.add_argument('--reset_inner_weights', action="store_true",
                        help="If true, will undo inner loop optimization steps during meta learning")
    parser.add_argument('--gradient_scale', type=float, default=1.0,
                        help="Scale the gradient during accumulation to prevent overflow/underflow")
    
    # Meta-learning base approaches
    parser.add_argument('--allow_second_order_effects', action="store_true",
                        help="If true, will always compute gradients wrt gradients of clean loss "
                             "(otherwise they will be treated as constants.)")
    parser.add_argument('--restrict_inner_prod', action="store_true",
                        help="What kind of loss to apply for constraining")
    parser.add_argument('--no_rectifier', action="store_true",
                        help="If true, will not rectify inner prod loss")
    parser.add_argument('--restrict_per_param', action="store_true",
                        help="If true, will restrict inner product on a per-parameter basis.")
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    args.n_gpu = torch.cuda.device_count()

    # Set seed
    set_seed(args)

    # Training
    if args.do_train:
        global_step, tr_loss = train(args)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == "__main__":
    main()
