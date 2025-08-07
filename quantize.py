import argparse, json
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from awq import AutoAWQForCausalLM

def parse_arguments():
    parser = argparse.ArgumentParser(description='Model quantization script')
    
    parser.add_argument('--model_id', type=str, required=True,
                       help='Path to the model directory')
    parser.add_argument('--save_dir', type=str, required=True,
                       help='Directory to save the quantized model')
    parser.add_argument('--recipe', type=str, default=None,
                       help='Path to the quantization recipe file')
    parser.add_argument('--compressor_type', type=str, default='llm_compressor',
                       choices=['llm_compressor', 'awq'],
                       help='Type of compressor to use')

    parser.add_argument('--dataset', type=str, default="HuggingFaceH4/ultrachat_200k",
                       help='Dataset name to use for calibration')
    parser.add_argument('--split', type=str, default="train_sft",
                       help='Dataset split to use')
    parser.add_argument('--column', type=str, default="messages",
                       help='Column name in the dataset')
    
    parser.add_argument('--num_calibration_samples', type=int, default=512,
                       help='Number of samples to use for calibration')
    parser.add_argument('--max_sequence_length', type=int, default=2048,
                       help='Maximum sequence length for tokenization')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, device_map="auto")
    if args.compressor_type == 'llm_compressor':
        model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="auto")
        ds = load_dataset(args.dataset, split=f"{args.split}[:{args.num_calibration_samples}]")
        ds = ds.shuffle(seed=42)
        
        def preprocess(example):
            return {"text": tokenizer.apply_chat_template(example[args.column], tokenize=False)}
        
        ds = ds.map(preprocess)
        
        def tokenize(sample):
            return tokenizer(sample["text"], padding=False, max_length=args.max_sequence_length, 
                            truncation=True, add_special_tokens=False)
        
        ds = ds.map(tokenize, remove_columns=ds.column_names)
        
        oneshot(model=model, recipe=args.recipe, dataset=ds, 
                max_seq_length=args.max_sequence_length, 
                num_calibration_samples=args.num_calibration_samples)
        model.save_pretrained(args.save_dir)
    else:
        model = AutoAWQForCausalLM.from_pretrained(args.model_id, device_map="auto")
        with open(args.recipe, 'r') as f:
            quant_config = json.load(f)

        model.quantize(tokenizer,
                calib_data=args.dataset,
                split=args.split,
                quant_config=quant_config,
                text_column=args.column,
                max_calib_samples=args.num_calibration_samples,
                max_calib_seq_len=args.max_sequence_length)

        model.save_quantized(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    
    print(f"Quantized model saved to: {args.save_dir}")

if __name__ == "__main__":
    main()
