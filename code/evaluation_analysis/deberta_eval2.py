from main import ask_from_llm
import pandas as pd
import argparse

def deberta_eval2(args):
    
    def llm_evaluate(row):
      prompt = row['prompt']
      response = row['response']
      prompt_with_response = f"Prompt: {prompt} Response : {response}"

      evaluation = ask_from_llm(prompt_with_response, eval_mode=True)

      if evaluation == "Yes":
         jailbroken = "1"
      elif evaluation == "No":
         jailbroken = "0"
      else:
         jailbroken = "Cannot determine"

      return jailbroken
    
    #Create a dataframe
    prompts_and_responses = pd.read_csv(args.data_path)

   # Ensure there are at least 2 columns (prompt and response)
    if len(prompts_and_responses.columns) < 2:
      raise ValueError("Input CSV must have at least 2 columns (prompt and response)")

    print(prompts_and_responses.head())

    prompts_and_responses['LLM Evaluation Eval2'] = prompts_and_responses.apply(llm_evaluate, axis=1)

    # Write the modified dataframe to a new CSV file, keeping the original order
    prompts_and_responses.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, required=True, help='data path to the csv file containing prompts and responses')
    parser.add_argument('--output_file', type=str, required=True, help='name of output file')
    # parser.add_argument('--retry_times', type=int, default=10, help='retry times when exception occurs')

    args = parser.parse_args()

    deberta_eval2(args)

'''
to run,
python deberta_eval2.py --data_path "res.csv" --output_file "eval3.csv"
'''         
