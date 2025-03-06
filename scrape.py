import requests
import pandas as pd
from datetime import datetime

def collect_poshmark_data(brand_query, max_items=100):
    """
    Collect detailed sold item data from Poshmark API
    """
    items = []
    base_url = "https://poshmark.com/vm-rest/posts"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36"
    }
    
    # Pagination parameters
    count_per_request = 48
    requests_needed = min((max_items + count_per_request - 1) // count_per_request, 10)  # Cap at 10 requests
    
    for i in range(requests_needed):
        url = f"{base_url}?request={{%22filters%22:{{%22department%22:%22All%22,%22inventory_status%22:[%22sold_out%22]}},%22query%22:%22{brand_query}%22,%22count%22:%22{count_per_request}%22,%22start%22:{i*count_per_request}}}&summarize=true"
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            if 'data' not in data:
                break
                
            for item in data['data']:
                # Extract detailed information
                title = item.get('title', 'N/A')
                brand = item.get('brand', 'Unknown')
                description = item.get('description', '')
                price = float(item.get('price', {}).get('amount', 0)) if item.get('price') else 0
                original_price = float(item.get('original_price', {}).get('amount', 0)) if item.get('original_price') else price
                size = item.get('size', 'Unknown')
                condition = item.get('condition', 'Unknown')
                department = item.get('department', 'Unknown')
                category = item.get('category', 'Unknown')
                created_at = item.get('created_at', '')
                sold_at = item.get('inventory', {}).get('status_changed_at', '')
                
                # Derived features
                color = extract_color(description)
                gender = extract_gender(description)
                item_type = classify_item(description)
                days_to_sell = calculate_days_to_sell(created_at, sold_at) if created_at and sold_at else -1
                
                items.append({
                    'title': title,
                    'brand': brand,
                    'color': color,
                    'gender': gender,
                    'type': item_type,
                    'price': price,
                    'original_price': original_price,
                    'size': size,
                    'condition': condition,
                    'department': department,
                    'category': category,
                    'created_at': created_at,
                    'sold_at': sold_at,
                    'days_to_sell': days_to_sell
                })
                
                if len(items) >= max_items:
                    break
                    
            if len(items) >= max_items:
                break
                
        except requests.RequestException as e:
            print(f"Error fetching data: {e}")
            break
    
    return pd.DataFrame(items)

def extract_color(description):
    """Extract color from description"""
    colors = ['red', 'blue', 'green', 'black', 'white', 'yellow', 'purple', 'pink', 'gray', 'brown']
    desc_lower = description.lower()
    for color in colors:
        if color in desc_lower:
            return color
    return 'Unknown'

def extract_gender(description):
    """Determine gender from description"""
    desc_lower = description.lower()
    if 'men' in desc_lower or 'male' in desc_lower:
        return 'Male'
    elif 'women' in desc_lower or 'female' in desc_lower or 'lady' in desc_lower:
        return 'Female'
    return 'Unisex'

def classify_item(description):
    """Classify item type from description"""
    desc_lower = description.lower()
    if 'shoe' in desc_lower or 'sneaker' in desc_lower or 'boot' in desc_lower:
        return 'Shoe'
    elif 'bag' in desc_lower or 'purse' in desc_lower or 'backpack' in desc_lower:
        return 'Bag'
    elif 'shirt' in desc_lower or 'top' in desc_lower or 'blouse' in desc_lower:
        return 'Shirt'
    elif 'jacket' in desc_lower or 'coat' in desc_lower:
        return 'Jacket'
    return 'Other'

def calculate_days_to_sell(created_at, sold_at):
    """Calculate days between listing and sale"""
    try:
        created = datetime.strptime(created_at, '%Y-%m-%dT%H:%M:%SZ')
        sold = datetime.strptime(sold_at, '%Y-%m-%dT%H:%M:%SZ')
        return (sold - created).days
    except (ValueError, TypeError):
        return -1

# Example usage
if __name__ == "__main__":
    df = collect_poshmark_data('nike')
    print(df.head())
    print(f"Collected {len(df)} items")