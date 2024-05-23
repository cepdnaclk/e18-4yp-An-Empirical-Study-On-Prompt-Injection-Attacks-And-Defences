import pandas as pd

def getting_data(file_to_output):
    """
    This function gets two csv file as data, and outputs a csv file with only prompts
    """
    jbTemplates_df = pd.read_csv('jbTemplates_example.csv')

    questions_df = pd.read_csv('questions_example.csv')

    # Add a key for merging
    jbTemplates_df['key'] = 1
    questions_df['key'] = 1

    # Cartesian product of A and B
    df_product = pd.merge(jbTemplates_df, questions_df, on='key').drop('key', axis=1)

    # Replace the placeholder text with the name
    df_product['text'] = df_product.apply(
        lambda x: x['text'].replace('[INSERT PROMPT HERE]', x['question']),
        axis=1
    )

    prompts_only = df_product['text']
    print(prompts_only.head())

    prompts_only = prompts_only.iloc[1:]

    # Export the DataFrame to a CSV file
    prompts_only.to_csv(file_to_output, index=False)  # index=False excludes the row index from the CSV