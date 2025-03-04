# search.py
import pandas as pd
from fuzzywuzzy import fuzz

def search_items(df, query):
    """
    Search for items in the DataFrame based on the query string.
    Supports partial matches and similar words.
    """
    query = query.lower().strip()
    
    def calculate_relevance(row):
        # Calculate relevance score based on multiple fields
        title_score = fuzz.partial_ratio(query, str(row['Clothes Title']).lower())
        desc_score = fuzz.partial_ratio(query, str(row['Clothes Description']).lower())
        dept_score = fuzz.partial_ratio(query, str(row['Department Name']).lower())
        class_score = fuzz.partial_ratio(query, str(row['Class Name']).lower())
        division_score = fuzz.partial_ratio(query, str(row['Division Name']).lower())
        
        # Weight the scores (adjust weights as needed)
        return (title_score * 0.3 + 
                desc_score * 0.3 + 
                dept_score * 0.15 + 
                class_score * 0.15 +
                division_score * 0.1)
    
    # Calculate relevance scores for all items
    df['relevance'] = df.apply(calculate_relevance, axis=1)
    
    # Filter items with relevance score above threshold
    relevant_items = df[df['relevance'] >= 50].sort_values('relevance', ascending=False)
    
    # Convert to list of dictionaries and remove the temporary relevance column
    results = relevant_items.drop('relevance', axis=1).to_dict('records')
    
    return results