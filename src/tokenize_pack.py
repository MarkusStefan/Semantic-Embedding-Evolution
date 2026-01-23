import multiprocessing
from pathlib import Path
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer


def pack_and_tokenize_dataset(
    save_dir: str = "data/pg19",
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    block_size: int = 128,
    subset: int | None = 500,
    batch_size: int = 100,
    num_proc: int | None = None,
    output_path: str | None = "data/pg19_processed_tokens",
):
    if num_proc is None:
        num_proc = multiprocessing.cpu_count()
    save_dir_path = Path(save_dir)
    if save_dir_path.exists():
        ds = load_from_disk(str(save_dir_path))
    else:
        ds = load_dataset("emozilla/pg19")
    if hasattr(ds, "keys"):
        try:
            ds = ds["train"]
        except Exception:
            pass
    if subset is not None:
        ds = ds.select(range(subset))
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    
    def group_texts(examples):
        tokenized_inputs = tokenizer(
            examples["text"],
            return_special_tokens_mask=True,
            truncation=False,
            add_special_tokens=False,
        )
        concatenated_examples = {k: sum(tokenized_inputs[k], []) for k in tokenized_inputs.keys()}
        total_length = len(concatenated_examples["input_ids"])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    cols_to_remove = ds.column_names if hasattr(ds, "column_names") else []
    lm_dataset = ds.map(
        group_texts,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=cols_to_remove,
    )
    if output_path is not None:
        lm_dataset.save_to_disk(output_path)
    return lm_dataset
