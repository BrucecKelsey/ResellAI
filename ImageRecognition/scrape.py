import requests
import pandas as pd
from datetime import datetime
import os
import json
import uuid
import urllib.parse
import time

def extract_scalar(value):
    """Extract a scalar value from a dict or return string if None."""
    if isinstance(value, dict):
        return str(value.get('value', 'Unknown') or value.get('name', 'Unknown'))
    return str(value) if value is not None else 'Unknown'

def extract_color(description):
    """Extract color from item description."""
    colors = ['red', 'blue', 'green', 'black', 'white', 'yellow', 'purple', 'pink', 'gray', 'brown']
    desc_lower = str(description).lower()
    for color in colors:
        if color in desc_lower:
            return color
    return 'Unknown'

def extract_gender(description):
    """Extract gender from item description."""
    desc_lower = str(description).lower()
    if 'men' in desc_lower or 'male' in desc_lower:
        return 'Male'
    elif 'women' in desc_lower or 'female' in desc_lower or 'lady' in desc_lower:
        return 'Female'
    return 'Unisex'

def classify_item(description, category, category_features):
    """Classify item type, focusing on shoe variants."""
    desc_lower = str(description).lower()
    cat_lower = str(category.get('display', '')).lower() if category else ''
    cat_features = [str(feat.get('display', '')).lower() for feat in category_features] if category_features else []

    if any(kw in desc_lower for kw in ['shoe', 'sneaker', 'boot', 'sandal', 'heel']):
        return 'Shoe'
    if 'shoes' in cat_lower or 'footwear' in cat_lower:
        return 'Shoe'
    if any('shoes' in feat or 'footwear' in feat for feat in cat_features):
        return 'Shoe'
    return None

def calculate_days_to_sell(created_at, sold_at):
    """Calculate days between creation and sale."""
    try:
        created = datetime.strptime(created_at, '%Y-%m-%dT%H:%M:%S%z')
        sold = datetime.strptime(sold_at, '%Y-%m-%dT%H:%M:%S%z')
        return (sold - created).days
    except (ValueError, TypeError) as e:
        print(f"Debug - Failed to parse timestamps: created_at={created_at}, sold_at={sold_at}, error={e}")
        return -1

def download_images(listing_id, photo_urls, output_dir="shoe_images"):
    """Download images and return a list of relative file paths."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_paths = []
    for i, photo_url in enumerate(photo_urls):
        if 'http' in photo_url:
            try:
                response = requests.get(photo_url, stream=True, timeout=10)
                response.raise_for_status()
                filename = f"listing_{listing_id}_photo_{i+1}.jpg"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                print(f"Downloaded {filepath}")
                image_paths.append(os.path.join("shoe_images", filename))  # Relative path
            except requests.RequestException as e:
                print(f"Error downloading {photo_url} for listing {listing_id}: {e}")
    return image_paths

def fetch_listing_details(listing_id):
    """Fetch full listing details to get multiple photos."""
    detail_url = f"https://poshmark.com/vm-rest/posts/{listing_id}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36"
    }
    try:
        response = requests.get(detail_url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching details for listing {listing_id}: {e}")
        return None

def collect_poshmark_data(brand_queries=None, category_filter=None, max_items=100):
    """
    Scrape Poshmark API for sold shoe items with multiple photos.
    
    Args:
        brand_queries (list or str): Single brand (str) or list of brands to query.
        category_filter (str): Category to filter by (e.g., 'Shoes').
        max_items (int): Maximum number of items to scrape (default 100).
    
    Returns:
        pd.DataFrame: DataFrame with scraped shoe items, photo URLs, and image paths.
    """
    items = []
    base_url = "https://poshmark.com/vm-rest/posts"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36"
    }
    
    count_per_request = 48
    requests_needed = min((max_items + count_per_request - 1) // count_per_request, 10)
    
    if brand_queries is None:
        brand_queries = [None]
    elif isinstance(brand_queries, str):
        brand_queries = [brand_queries]
    
    for brand_query in brand_queries:
        items_collected_for_brand = 0
        for i in range(requests_needed):
            filters = {
                "department": "All",
                "inventory_status": ["sold_out"],
                "query": f"{brand_query} shoes" if brand_query else None
            }
            filters = {k: v for k, v in filters.items() if v is not None}
            
            request_dict = {
                "filters": filters,
                "count": str(count_per_request),
                "start": i * count_per_request
            }
            
            request_json = json.dumps(request_dict)
            request_encoded = urllib.parse.quote(request_json, safe='')
            url = f"{base_url}?request={request_encoded}&summarize=true"
            
            print(f"Fetching data for brand '{brand_query}' at offset {i*count_per_request}: {url}")
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if 'data' not in data or not data['data']:
                    print(f"No more data for {brand_query} at offset {i*count_per_request}")
                    break
                
                for item in data['data']:
                    description = item.get('description', '')
                    category = item.get('category_v2', {})
                    category_features = item.get('category_features', [])
                    item_type = classify_item(description, category, category_features)
                    if item_type != 'Shoe':
                        print(f"Skipping item '{item.get('title', 'N/A')}' - not a shoe (category: {category.get('display', 'N/A')})")
                        continue
                    
                    listing_id = item.get('id', str(uuid.uuid4()))
                    title = item.get('title', 'N/A')
                    brand = item.get('brand', 'Unknown')
                    
                    price_data = item.get('price_amount', {})
                    price = float(price_data.get('val', 0)) if isinstance(price_data, dict) else float(price_data or 0)
                    
                    orig_price_data = item.get('original_price_amount', {})
                    original_price = float(orig_price_data.get('val', 0)) if isinstance(orig_price_data, dict) else float(orig_price_data or price)
                    
                    size = extract_scalar(item.get('size', 'Unknown'))
                    condition = extract_scalar(item.get('condition', 'Unknown'))
                    department = extract_scalar(item.get('department', {}).get('display', 'Unknown'))
                    category = extract_scalar(item.get('category_v2', {}).get('display', 'Unknown'))
                    
                    created_at = item.get('created_at', '')
                    sold_at = item.get('inventory', {}).get('status_changed_at', '')
                    days_to_sell = calculate_days_to_sell(created_at, sold_at) if created_at and sold_at else -1
                    
                    # Fetch full listing details to get multiple photos
                    detail_data = fetch_listing_details(listing_id)
                    if detail_data and 'data' in detail_data:
                        detail_item = detail_data['data']
                        # Try multiple fields for photos
                        picture_urls = detail_item.get('picture_urls', [])
                        photo_urls = []
                        if picture_urls:
                            for pic in picture_urls:
                                url = pic.get('url_large', pic.get('url', 'No Photo'))
                                if url != 'No Photo' and 'http' in url:
                                    photo_urls.append(url)
                        else:
                            # Fallback to cover photo or picture_url
                            cover_photo = detail_item.get('cover_photo', {})
                            if cover_photo and isinstance(cover_photo, dict):
                                url = cover_photo.get('url', 'No Photo')
                                if url != 'No Photo' and 'http' in url:
                                    photo_urls.append(url)
                            picture_url = detail_item.get('picture_url', 'No Photo')
                            if picture_url != 'No Photo' and 'http' in picture_url:
                                photo_urls.append(picture_url)
                    else:
                        # Fallback to search API data
                        picture_urls = item.get('picture_urls', [])
                        photo_urls = []
                        if picture_urls:
                            for pic in picture_urls:
                                url = pic.get('url_large', pic.get('url', 'No Photo'))
                                if url != 'No Photo' and 'http' in url:
                                    photo_urls.append(url)
                        else:
                            picture_url = item.get('picture_url', 'No Photo')
                            if picture_url != 'No Photo' and 'http' in picture_url:
                                photo_urls.append(picture_url)
                    
                    if not photo_urls:
                        photo_urls = ['No Photo']
                    
                    # Download images and get relative paths
                    image_paths = download_images(listing_id, photo_urls) if photo_urls[0] != 'No Photo' else []
                    
                    color = extract_color(description)
                    gender = extract_gender(description)
                    
                    items.append({
                        'listing_id': listing_id,
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
                        'days_to_sell': days_to_sell,
                        'photo_urls': json.dumps(photo_urls),
                        'image_paths': json.dumps(image_paths)  # Store relative paths to images
                    })
                    
                    items_collected_for_brand += 1
                    print(f"Collected item: {title} (Brand: {brand}, Listing ID: {listing_id}, Photos: {len(photo_urls)})")
                    
                    if len(items) >= max_items:
                        print(f"Reached max_items ({max_items})")
                        break
                
                if len(items) >= max_items:
                    break
                
                if items_collected_for_brand == 0 and i == requests_needed - 1:
                    print(f"No shoe items found for {brand_query} after {requests_needed} requests. Moving to next brand.")
                    break
                
            except requests.RequestException as e:
                print(f"Error fetching data for {brand_query} at offset {i*count_per_request}: {e}")
                break
            except ValueError as e:
                print(f"Error processing data for {brand_query}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error for {brand_query}: {e}")
                continue
            
            time.sleep(2)
    
    if not items:
        print("No items collected. Check API response or filters.")
        return pd.DataFrame()
    
    df = pd.DataFrame(items)
    numeric_cols = ['price', 'original_price', 'days_to_sell']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1)
    
    print(f"Total items collected: {len(df)}")
    return df

if __name__ == "__main__":
    brands = ['nike', 'converse', 'adidas', 'puma', 'vans', 'new_balance', 'jordan', 
              'under_armour', 'timberland', 'skechers']
    df = collect_poshmark_data(brand_queries=brands, max_items=100)
    if not df.empty:
        df.to_csv("shoe_data_with_photos.csv", index=False)
        print(df.head())
    else:
        print("No data collected. Please review logs for errors.")