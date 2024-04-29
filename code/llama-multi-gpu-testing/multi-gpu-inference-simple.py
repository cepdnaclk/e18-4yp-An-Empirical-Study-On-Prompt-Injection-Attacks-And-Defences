from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
from statistics import mean
import torch, time, json
import os

accelerator = Accelerator()

prompts_all=[
    "The King is dead. Long live the Queen.",
    "Once there were four children whose names were Peter, Susan, Edmund, and Lucy.",
    "The story so far: in the beginning, the universe was created.",
    "It was a bright cold day in April, and the clocks were striking thirteen.",
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
    "The sweat wis lashing oafay Sick Boy; he wis trembling.",
    "124 was spiteful. Full of Baby's venom.",
    "As Gregor Samsa awoke one morning from uneasy dreams he found himself transformed in his bed into a gigantic insect.",
    "I write this sitting in the kitchen sink.",
    "We were somewhere around Barstow on the edge of the desert when the drugs began to take hold.",
] * 2

# load a base model and tokenizer
model_path="meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_path,    
    device_map={"": accelerator.process_index},
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)   

# sync GPUs and start the timer
accelerator.wait_for_everyone()
start=time.time()

# divide the prompt list onto the available GPUs 
with accelerator.split_between_processes(prompts_all) as prompts:
    # store output of generations in dict
    results=dict(outputs=[], num_tokens=0)
    
    # have each GPU do inference, prompt by prompt
    for prompt in prompts:
        prompt_tokenized=tokenizer(prompt, return_tensors="pt").to("cuda")
        # print(prompt_tokenized)
        output_tokenized = model.generate(prompt_tokenized.input_ids, max_new_tokens=100)
        
        

        # remove prompt from output 
        # output_tokenized=output_tokenized[len(prompt_tokenized["input_ids"][0]):]

        # store outputs and number of tokens in result{}
        results["outputs"].append( tokenizer.batch_decode(output_tokenized, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] )
        results["num_tokens"] += len(output_tokenized)
    
    results=[ results ] # transform to list, otherwise gather_object() will not collect correctly

# collect results from all the GPUs
results_gathered=gather_object(results)

if accelerator.is_main_process:
    timediff = time.time() - start
    num_tokens = sum([r["num_tokens"] for r in results_gathered])
    output_str = f"tokens/sec: {num_tokens//timediff}, time {timediff}, total tokens {num_tokens}, total prompts {len(prompts_all)}"

    # Create a directory for output if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Write the output to a text file
    output_file = os.path.join(output_dir, "results.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for outStr in results_gathered:
            print(outStr)
            f.write(str(outStr))
        f.write(f"TIME: {timediff}")

    print(f"Results written to {output_file}")