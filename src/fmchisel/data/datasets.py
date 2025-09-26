from abc import ABC, abstractmethod

import datasets
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from fmchisel.config import DataLoadingConfig
from fmchisel.data.collator import DataCollatorForCompletionOnlyLM

_RETAIN_COLUMNS = {"input_ids", "attention_mask", "labels"}


CNN_RESPONSE_TEMPLATE = " <Highlight> "


class DataModule(pl.LightningDataModule, ABC):
    def __init__(self, tokenizer: AutoTokenizer, data_load_config: DataLoadingConfig):
        super().__init__()
        self.data_name = data_load_config.dataset
        self.tokenizer = tokenizer
        self.data_path = data_load_config.data_path
        self.max_length = data_load_config.max_length
        self.batch_size = data_load_config.batch_size
        self.n_train = data_load_config.n_train
        self.n_val = data_load_config.n_val
        self.return_prompt_input_ids = data_load_config.return_prompt_input_ids

    @abstractmethod
    def formatting_func(self, example):
        pass

    def tokenize(self, example):
        outputs = self.tokenizer(
            self.formatting_func(example),
            truncation=True,
            padding=False,
            max_length=self.max_length,
        )
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }

    @abstractmethod
    def setup(self, stage) -> None:
        self.train_dataset = self.dataset["train"].map(
            self.tokenize,
            remove_columns=list(set(self.dataset["train"].column_names) - _RETAIN_COLUMNS),
            batched=True,
            batch_size=1,
        )
        self.val_dataset = self.dataset["test"].map(
            self.tokenize,
            remove_columns=list(set(self.dataset["test"].column_names) - _RETAIN_COLUMNS),
            batched=True,
            batch_size=1,
        )
        self.dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator,
        )


class CNNModule(DataModule):
    def __init__(self, tokenizer: AutoTokenizer, data_load_config: DataLoadingConfig):
        super().__init__(tokenizer, data_load_config)
        response_prompt = tokenizer.encode(CNN_RESPONSE_TEMPLATE, add_special_tokens=False)
        self.collator = DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer,
            response_template=response_prompt,
            pad_to_multiple_of=16,
            return_prompt_input_ids=self.return_prompt_input_ids,
        )

    def formatting_func(self, example):
        output = "Given a text, please give highlights.\n\n"
        output += f"TEXT: {example['article']}\n"
        output += f" {CNN_RESPONSE_TEMPLATE} "
        output += f"{example['highlights']} "
        return [output]

    def setup(self, stage) -> None:
        self.dataset = (
            datasets.load_dataset(path=self.data_path)
            if self.data_path
            else datasets.load_dataset("cnn_dailymail", "3.0.0")
        )
        self.dataset["train"] = self.dataset["train"].select(range(self.n_train))
        self.dataset["test"] = self.dataset["test"].select(range(self.n_val))
        super().setup(stage)

class OpenThoughtsModule(DataModule):
    def __init__(self, tokenizer: AutoTokenizer, 
                 data_load_config: DataLoadingConfig):
        super().__init__(tokenizer, data_load_config)
        
        self.tokenizer = tokenizer
        # Strategy for keeping exactly 50% of tokens. None disables this behaviour.
        self.half_keep_strategy = getattr(data_load_config, "half_keep_strategy", None)
        self.truncate_after_think_end_token = getattr(data_load_config, "truncate_after_think_end_token", False)
        if self.truncate_after_think_end_token:
            self.think_end_token_ids = self.tokenizer.encode(data_load_config.cot_end_token, add_special_tokens=False)
        else:
            self.think_end_token_ids = None
        
        self.collator = DataCollatorForCompletionOnlyLM(
                tokenizer=tokenizer,
                response_template=data_load_config.response_template,
                pad_to_multiple_of=16,
                return_prompt_input_ids=self.return_prompt_input_ids,
                section_inclusion=data_load_config.section_inclusion,
                random_mask_ratio=data_load_config.random_mask_ratio,
                included_first_x_percent=data_load_config.included_first_x_percent,
                half_keep_strategy=data_load_config.half_keep_strategy,
                cot_start_token = data_load_config.cot_start_token,
                cot_end_token = data_load_config.cot_end_token,
                is_reasoning_llm = data_load_config.is_reasoning_llm,
            )
        # Store truncation ratio for later use in tokenization
        self.included_first_x_percent = data_load_config.included_first_x_percent

    def tokenize(self, example):
        """Tokenize and *truncate* each example *before* padding so that
        the final batch tensors are genuinely shorter.

        The parent DataModule.tokenize() keeps full-length tokenised
        sequences and relies on the collator to mask out unwanted
        positions.  That wastes compute because the model still runs
        attention over the masked tokens.

        Here we truncate the sequence to the first
        `included_first_x_percent` tokens (ratio in \[0,1\]).  By
        shortening *before* the DataLoader stacks examples into a batch
        we ensure that the batch's `max_seq_len` really shrinks, giving
        proportional speed-ups during training.
        """

        # 1. Build the chat-templated text exactly like the default logic.
        formatted_texts = self.formatting_func(example)  # returns List[str]

        # 2. Tokenise *without* truncation or padding so we get the full length.
        outputs = self.tokenizer(
            formatted_texts,
            truncation=False,
            padding=False,
            max_length=None,
        )

        input_ids      = outputs["input_ids"][0]        # a len-1 list [[token_id1, token_id2, ...]]
        attention_mask = outputs["attention_mask"][0]    # a len-1 list [[1, 1, ...]]

        if self.truncate_after_think_end_token and self.think_end_token_ids:
            think_end_ids = self.think_end_token_ids
            # find first occurrence of think_end_ids and truncate
            for i in range(len(input_ids) - len(think_end_ids) + 1):
                if input_ids[i : i + len(think_end_ids)] == think_end_ids:
                    truncation_point = i + len(think_end_ids)
                    # print(f'Rightmost {len(input_ids) - truncation_point} of tokens will be truncated out of original {len(input_ids)} token')
                    input_ids = input_ids[:truncation_point]
                    attention_mask = attention_mask[:truncation_point]                    
                    break
        # print(f'outputs["input_ids"]: {outputs["input_ids"]}')
        # print(f'outputs["attention_mask"]: {outputs["attention_mask"]}')
        # print(f'outputs["input_ids"][0]: {outputs["input_ids"][0]}')
        # print(f'outputs["attention_mask"][0]: {outputs["attention_mask"][0]}')

        # ------------------------------------------------------------------
        # Token subsampling logic
        # ------------------------------------------------------------------
        if self.half_keep_strategy is not None:
            # Always keep exactly 50% tokens according to selected strategy.
            total_len = len(input_ids)
            if total_len == 0:
                # Defensive: shouldn't happen but guard against zero-length.
                kept_ids = input_ids
                kept_mask = attention_mask
            else:
                half_len = max(1, int(total_len * 0.5))  # ensure at least 1 token
                if self.half_keep_strategy == "left":
                    start = 0
                elif self.half_keep_strategy == "middle":
                    start = max(0, (total_len - half_len) // 2)
                elif self.half_keep_strategy == "right":
                    start = max(0, total_len - half_len)
                else:
                    raise ValueError(f"Unknown half_keep_strategy: {self.half_keep_strategy}")
                end = start + half_len
                kept_ids = input_ids[start:end]
                kept_mask = attention_mask[start:end]
            input_ids = kept_ids
            attention_mask = kept_mask
        elif 0.0 < self.included_first_x_percent < 1.0:
            keep = int(len(input_ids) * self.included_first_x_percent)
            # ensure at least 1 token is kept to avoid empty sequences
            keep = max(1, keep) #defensive
            input_ids      = input_ids[:keep]
            attention_mask = attention_mask[:keep]

        return {
            "input_ids": [input_ids],        # keep nested list structure
            "attention_mask": [attention_mask],
        }

    def formatting_func(self, example):
        '''Ignore sys msg, just use conversations'''
        conversations = example["conversations"]
        if isinstance(conversations, str):
            print('Note! Conversations are in str data type, not list')
            conversations = json.loads(conversations)
        
        # print(f'type(conversations): {type(conversations)}')
        # print(f'len(conversations): {len(conversations)}')
        # print(f'conversations: {conversations}')
        # Build prompt from user and assistant messages
        messages = []
        for turn in conversations[0]: # `conversations` is a len-1 list
            content = turn["value"]
            if turn["from"] == "user":
                messages.append({"role": "user", "content": content})
            elif turn["from"] == "assistant":
                # replace <|begin_of_solution|> and <|end_of_solution|> with empty strings
                content = content.replace("<|begin_of_solution|>\n\n", '').replace("\n\n<|end_of_solution|>", '')
                # replace <|begin_of_thought|> and <|end_of_thought|> with <think> and </think>, respectively
                content = content.replace("<|begin_of_thought|>", "<think>").replace("<|end_of_thought|>", "</think>")
                messages.append({"role": "assistant", "content": content})
            else:
                raise ValueError(f"Unknown role {turn['from']} in conversation turn: {turn}")
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
        )
        # print(f'text in OpenThoughtsModule formatting_func: {text}')
        return [text]

    def setup(self, stage) -> None:
        raw = datasets.load_dataset(self.data_path)
        # see ella/src/ella/data/check_ex_num_of_tokens.py
        try:
            from .included_indexes_openthoughts114k import indexes_of_ex_with_less_than_4k_tokens
        except ImportError:
            from ella.data.included_indexes_openthoughts114k import indexes_of_ex_with_less_than_4k_tokens

        # Case A: both train & test splits exist
        if "train" in raw and "test" in raw:
            train_split = raw["train"].select(indexes_of_ex_with_less_than_4k_tokens).select(range(self.n_train))
            test_split  = raw["test"].select(indexes_of_ex_with_less_than_4k_tokens).select(range(self.n_val))

        # Case B: only train exists, create test via train_test_split
        else:
            splits     = raw["train"].select(indexes_of_ex_with_less_than_4k_tokens).train_test_split(test_size=self.n_val, seed=42)
            train_split = splits["train"].select(range(self.n_train))
            test_split  = splits["test"]

        self.dataset = {"train": train_split, "test": test_split}
        super().setup(stage)


class BespokeStratosModule(DataModule):
    # bespokelabs/Bespoke-Stratos-17k
    def __init__(self, tokenizer: AutoTokenizer, 
                 data_load_config: DataLoadingConfig):
        super().__init__(tokenizer, data_load_config)
        
        self.tokenizer = tokenizer
        # Strategy for keeping exactly 50% of tokens. None disables this behaviour.
        self.half_keep_strategy = getattr(data_load_config, "half_keep_strategy", None)
        self.truncate_after_think_end_token = getattr(data_load_config, "truncate_after_think_end_token", False)
        if self.truncate_after_think_end_token:
            self.think_end_token_ids = self.tokenizer.encode(data_load_config.cot_end_token, add_special_tokens=False)
        else:
            self.think_end_token_ids = None
        
        self.collator = DataCollatorForCompletionOnlyLM(
                tokenizer=tokenizer,
                response_template=data_load_config.response_template,
                pad_to_multiple_of=16,
                return_prompt_input_ids=self.return_prompt_input_ids,
                section_inclusion=data_load_config.section_inclusion,
                random_mask_ratio=data_load_config.random_mask_ratio,
                included_first_x_percent=data_load_config.included_first_x_percent,
                half_keep_strategy=data_load_config.half_keep_strategy,
                cot_start_token = data_load_config.cot_start_token,
                cot_end_token = data_load_config.cot_end_token,
                is_reasoning_llm = data_load_config.is_reasoning_llm,
            )
        # Store truncation ratio for later use in tokenization
        self.included_first_x_percent = data_load_config.included_first_x_percent

    def tokenize(self, example):
        """Tokenize and *truncate* each example *before* padding so that
        the final batch tensors are genuinely shorter.

        The parent DataModule.tokenize() keeps full-length tokenised
        sequences and relies on the collator to mask out unwanted
        positions.  That wastes compute because the model still runs
        attention over the masked tokens.

        Here we truncate the sequence to the first
        `included_first_x_percent` tokens (ratio in \[0,1\]).  By
        shortening *before* the DataLoader stacks examples into a batch
        we ensure that the batch's `max_seq_len` really shrinks, giving
        proportional speed-ups during training.
        """

        # 1. Build the chat-templated text exactly like the default logic.
        formatted_texts = self.formatting_func(example)  # returns List[str]

        # 2. Tokenise *without* truncation or padding so we get the full length.
        outputs = self.tokenizer(
            formatted_texts,
            truncation=False,
            padding=False,
            max_length=None,
        )

        input_ids      = outputs["input_ids"][0]        # a len-1 list [[token_id1, token_id2, ...]]
        attention_mask = outputs["attention_mask"][0]    # a len-1 list [[1, 1, ...]]

        if self.truncate_after_think_end_token and self.think_end_token_ids:
            think_end_ids = self.think_end_token_ids
            # find first occurrence of think_end_ids and truncate
            for i in range(len(input_ids) - len(think_end_ids) + 1):
                if input_ids[i : i + len(think_end_ids)] == think_end_ids:
                    truncation_point = i + len(think_end_ids)
                    print(f'Rightmost {len(input_ids) - truncation_point} of tokens will be truncated out of original {len(input_ids)} token')
                    input_ids = input_ids[:truncation_point]
                    attention_mask = attention_mask[:truncation_point]                    
                    break
        # print(f'outputs["input_ids"]: {outputs["input_ids"]}')
        # print(f'outputs["attention_mask"]: {outputs["attention_mask"]}')
        # print(f'outputs["input_ids"][0]: {outputs["input_ids"][0]}')
        # print(f'outputs["attention_mask"][0]: {outputs["attention_mask"][0]}')

        # ------------------------------------------------------------------
        # Token subsampling logic
        # ------------------------------------------------------------------
        if self.half_keep_strategy is not None:
            # Always keep exactly 50% tokens according to selected strategy.
            total_len = len(input_ids)
            if total_len == 0:
                # Defensive: shouldn't happen but guard against zero-length.
                kept_ids = input_ids
                kept_mask = attention_mask
            else:
                half_len = max(1, int(total_len * 0.5))  # ensure at least 1 token
                if self.half_keep_strategy == "left":
                    start = 0
                elif self.half_keep_strategy == "middle":
                    start = max(0, (total_len - half_len) // 2)
                elif self.half_keep_strategy == "right":
                    start = max(0, total_len - half_len)
                else:
                    raise ValueError(f"Unknown half_keep_strategy: {self.half_keep_strategy}")
                end = start + half_len
                kept_ids = input_ids[start:end]
                kept_mask = attention_mask[start:end]
            input_ids = kept_ids
            attention_mask = kept_mask
        elif 0.0 < self.included_first_x_percent < 1.0:
            keep = int(len(input_ids) * self.included_first_x_percent)
            # ensure at least 1 token is kept to avoid empty sequences
            keep = max(1, keep) #defensive
            input_ids      = input_ids[:keep]
            attention_mask = attention_mask[:keep]

        return {
            "input_ids": [input_ids],        # keep nested list structure
            "attention_mask": [attention_mask],
        }

    def formatting_func(self, example):
        '''Ignore sys msg, just use conversations'''
        conversations = example["conversations"]
        if isinstance(conversations, str):
            print('Note! Conversations are in str data type, not list')
            conversations = json.loads(conversations)
        
        # print(f'type(conversations): {type(conversations)}')
        # print(f'len(conversations): {len(conversations)}')
        # print(f'conversations: {conversations}')
        # Build prompt from user and assistant messages
        messages = []
        for turn in conversations[0]: # `conversations` is a len-1 list
            content = turn["value"]
            if turn["from"] == "user":
                messages.append({"role": "user", "content": content})
            elif turn["from"] == "assistant":
                # replace <|begin_of_solution|> and <|end_of_solution|> with empty strings
                content = content.replace("<|begin_of_solution|>\n\n", '').replace("\n\n<|end_of_solution|>", '')
                # replace <|begin_of_thought|> and <|end_of_thought|> with <think> and </think>, respectively
                content = content.replace("<|begin_of_thought|>", "<think>").replace("<|end_of_thought|>", "</think>")
                messages.append({"role": "assistant", "content": content})
            else:
                raise ValueError(f"Unknown role {turn['from']} in conversation turn: {turn}")
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
        )
        # print(f'text in OpenThoughtsModule formatting_func: {text}')
        return [text]

    def setup(self, stage) -> None:
        raw = datasets.load_dataset(self.data_path)
        # see ella/src/ella/data/check_ex_num_of_tokens.py
        try:
            from .included_indexes_bespokeStratos17k import indexes_of_ex_with_less_than_4k_tokens_bespoke_stratos
        except ImportError:
            from ella.data.included_indexes_bespokeStratos17k import indexes_of_ex_with_less_than_4k_tokens_bespoke_stratos

        # Case A: both train & test splits exist
        if "train" in raw and "test" in raw:
            train_split = raw["train"].select(indexes_of_ex_with_less_than_4k_tokens_bespoke_stratos).select(range(self.n_train))
            test_split  = raw["test"].select(indexes_of_ex_with_less_than_4k_tokens_bespoke_stratos).select(range(self.n_val))

        # Case B: only train exists, create test via train_test_split
        else:
            splits     = raw["train"].select(indexes_of_ex_with_less_than_4k_tokens_bespoke_stratos).train_test_split(test_size=self.n_val, seed=42)
            train_split = splits["train"].select(range(self.n_train))
            test_split  = splits["test"]

        self.dataset = {"train": train_split, "test": test_split}
        super().setup(stage)
