from datasets import Dataset
from dschat.utils.data.data_utils import create_prompt_dataset
from transformers import AutoTokenizer
from dschat.utils.utils import load_hf_tokenizer

import argparse

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    # parser.add_argument('--data_path',
    #                     nargs='*',
    #                     default=['./data/chem_data/orderly_train'],
    #                     help='Path to the training dataset. Accepted format:'
    #                     '1) a single data path, 2) multiple datasets in the'
    #                     'form: dataset1-path dataset2-path ...')
    # parser.add_argument(
    #     '--data_output_path',
    #     type=str,
    #     default='./tmp/data_files/',
    #     help=
    #     'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    # )
    # parser.add_argument("--local_rank",
    #                     type=int,
    #                     default=-1,
    #                     help="local_rank for distributed training on gpus")
    # parser.add_argument('--data_split',
    #                     type=str,
    #                     default='2,4,4',
    #                     help='Comma-separated list of proportions for training'
    #                     'phase 1, 2, and 3 data. For example the split `6,2,2`'
    #                     'will use 60%% of data for phase 1, 20%% for phase 2'
    #                     'and 20%% for phase 3.')
    # parser.add_argument(
    #     '--sft_only_data_path',
    #     nargs='*',
    #     default=[],
    #     help='Path to the dataset for only using in SFT phase.')
    # parser.add_argument(
    #     "--max_seq_len",
    #     type=int,
    #     default=512,
    #     help="The maximum sequence length.",
    # )
    # parser.add_argument(
    #     "--add_eot_token",
    #     action='store_true',
    #     help="Add <|endoftext|> as additional special token to tokenizer")
    # args =  parser.parse_args()
    # args.seed = 1234
    
    # args.end_of_conversation_token = "<|endoftext|>"
    # additional_special_tokens = args.end_of_conversation_token if args.add_eot_token else None
    model_name_or_path = '/data/yanyuliang/Code/got/hf_models/llama2/llama2-7b-chat'
    additional_special_tokens = "<|endoftext|>"
    
    tokenizer = load_hf_tokenizer(model_name_or_path,
                                fast_tokenizer=True,
                                add_special_tokens=additional_special_tokens)
    # tokenizer = AutoTokenizer.from_pretrained('/data/yanyuliang/Code/got/hf_models/llama2/llama2-7b-chat/')
    
    dataset = Dataset.load_from_disk('./data/chem_data/orderly_train_with_graph_test')
    # dataset_test = dataset.select(range(0, 10))
    print(dataset)
    
    # Prepare the data
    train_phase = 1
    local_rank = -1
    data_path = './data/chem_data/orderly_train_with_graph_test'
    data_split = '2,4,4'
    data_output_path = './tmp/data_files/'
    seed = 1234
    max_seq_len = 512
    sft_only_data_path = []
    
    train_dataset, eval_dataset = create_prompt_dataset(
        local_rank,
        data_path,
        data_split,
        data_output_path,
        train_phase,
        seed,
        tokenizer,
        max_seq_len,
        sft_only_data_path)