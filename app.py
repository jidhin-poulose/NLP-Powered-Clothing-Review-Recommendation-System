# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from search import search_items
from ml_model import generate_recommendation

app = Flask(__name__)

# Load the CSV data
df = pd.read_csv('data/assignment3_II.csv')

def group_items(df):
    """Group items by unique product identifiers"""
    grouped = df.groupby(['Division Name', 'Department Name', 'Class Name', 
                         'Clothes Title', 'Clothes Description']).agg({
        'Clothing ID': list,
        'Review Text': list,
        'Rating': list,
        'Recommended IND': list,
        'Positive Feedback Count': list,
    }).reset_index()
    
    # Add summary statistics
    grouped['avg_rating'] = grouped['Rating'].apply(lambda x: sum(x) / len(x) if x else 0)
    grouped['review_count'] = grouped['Review Text'].apply(len)
    grouped['recommend_percent'] = grouped['Recommended IND'].apply(
        lambda x: (sum(x) / len(x) * 100) if x else 0
    )
    
    return grouped.to_dict('records')

@app.route('/')
def home():
    # Get grouped items for the home page
    grouped_df = group_items(df)
    sample_items = pd.DataFrame(grouped_df).sample(n=min(6, len(grouped_df))).to_dict('records')
    return render_template('home.html', items=sample_items)

@app.route('/search')
def search():
    query = request.args.get('query', '')
    if query:
        # Search in relevant columns
        filtered_df = search_items(df, query)
        # Group the filtered results
        items = group_items(pd.DataFrame(filtered_df))
        num_results = len(items)
    else:
        items = []
        num_results = 0
    
    return render_template('search_results.html', 
                         items=items, 
                         query=query, 
                         num_results=num_results)

@app.route('/item/<path:clothes_title>')
def item_details(clothes_title):
    # Get all reviews for this item
    item_data = df[df['Clothes Title'] == clothes_title].to_dict('records')
    
    if not item_data:
        return "Item not found", 404
        
    # Get the basic item information from the first record
    item_info = {
        'Clothes Title': clothes_title,
        'Clothes Description': item_data[0]['Clothes Description'],
        'Division Name': item_data[0]['Division Name'],
        'Department Name': item_data[0]['Department Name'],
        'Class Name': item_data[0]['Class Name'],
    }
    
    # Calculate summary statistics
    reviews = item_data
    summary = {
        'avg_rating': sum(r['Rating'] for r in reviews) / len(reviews),
        'review_count': len(reviews),
        'recommend_count': sum(1 for r in reviews if r['Recommended IND'] == 1),
        'positive_feedback_total': sum(r['Positive Feedback Count'] for r in reviews),
    }
    summary['recommend_percent'] = (summary['recommend_count'] / len(reviews) * 100)
    
    return render_template('item_details.html', 
                         item=item_info, 
                         reviews=reviews,
                         summary=summary)

@app.route('/create_review/<path:clothes_title>', methods=['GET', 'POST'])
def create_review(clothes_title):
    if request.method == 'POST':
        review_text = request.form['review_text']
        rating = int(request.form['rating'])
        
        try:
            # Generate recommendation using ML model
            recommendation = generate_recommendation(review_text)
            print(f"Generated recommendation: {recommendation}")
            return jsonify({
                'success': True,
                'recommendation': bool(recommendation)
            })
        except Exception as e:
            print(f"Error in create_review: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Error generating recommendation'
            }), 500
    
    # GET request - show the review form
    item_data = df[df['Clothes Title'] == clothes_title].iloc[0].to_dict()
    return render_template('create_review.html', item=item_data)

if __name__ == '__main__':
    app.run(debug=True)