from main import ask_from_llm
import pandas as pd
import argparse

def deberta_eval2(args):
    #Create a dataframe
    prompts_and_responses = pd.read_csv(args.data_path)

   # Ensure there are at least 2 columns (prompt and response)
    if len(prompts_and_responses.columns) < 2:
      raise ValueError("Input CSV must have at least 2 columns (prompt and response)")
   
    # Add a new column with empty strings (using '')
    prompts_and_responses['LLM Evaluation Eval2'] = ''

    print(prompts_and_responses.head())

    for row in prompts_and_responses.interrow():
      print(row)
      prompt = row['prompt'] 
      response = row['response']
      prompt_with_response = f"Prompt: {prompt} Response : {response}"

      evaluation = ask_from_llm(prompt_with_response, eval_mode=True)

      if evaluation == "Yes":
         row['LLM Evaluation Eval2'] = 1
      elif evaluation == "No":
         row['LLM Evaluation Eval2'] = 0
      else:
         row['LLM Evaluation Eval2'] = "None"

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
