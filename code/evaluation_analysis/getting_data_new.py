import pandas as pd
import re
import unicodedata
import string

def cleaning_data(row):
    text = row['generation']
    
    text = str(text)
    text = re.sub(r'\n', '', text)
    text = unicodedata.normalize('NFKD', text)  # Normalize for better unicode handling
    
    allowed_chars = set(string.printable)  # Start with all printable characters
    text = ''.join([char for char in text if char in allowed_chars]).strip()
     
    # Reduce extra whitespace
    text = ' '.join(text.split()) 
    
    return text
    
       
def getting_data_new(file_to_input, file_to_output):
    """
    This function gets a csv file, and outputs a csv file with only prompts and responses
    """
    input_df = pd.read_csv(file_to_input)

    output_df = input_df.iloc[:, :2]
    
    output_df['cleaned text'] = output_df.apply(cleaning_data, axis=1)

    # Export the DataFrame to a CSV file
    output_df.to_csv(file_to_output, index=False)  # index=False excludes the row index from the CSV
    print("Done!")
    
    
getting_data_new("e18-4yp-An-Empirical-Study-On-Prompt-Injection-Attacks-And-Defences/Evaluation-methods/text_behaviours.csv", "e18-4yp-An-Empirical-Study-On-Prompt-Injection-Attacks-And-Defences/Evaluation-methods/output_new.csv")