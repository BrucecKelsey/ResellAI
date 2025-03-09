import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from scrape import collect_poshmark_data
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class ResellMarketAnalyzer:
    def __init__(self, data_file="resell_data.csv"):
        self.data_file = data_file
        self.items_df = self.load_data()
        self.label_encoders = {}
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.median_days = None
        
    def load_data(self):
        """Load existing data from CSV, ensuring numeric columns are floats."""
        if os.path.exists(self.data_file):
            try:
                df = pd.read_csv(self.data_file, dtype=str)
                for col in ['price', 'original_price', 'days_to_sell']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                return df
            except Exception as e:
                print(f"Error loading {self.data_file}: {e}")
                return pd.DataFrame(columns=[
                    'title', 'brand', 'color', 'gender', 'type', 'price',
                    'original_price', 'size', 'condition', 'department',
                    'category', 'days_to_sell'
                ])
        return pd.DataFrame(columns=[
            'title', 'brand', 'color', 'gender', 'type', 'price',
            'original_price', 'size', 'condition', 'department',
            'category', 'days_to_sell'
        ])
        
    def save_data(self):
        """Save current items_df to CSV with error handling."""
        try:
            self.items_df.to_csv(self.data_file, index=False)
        except PermissionError as e:
            print(f"Permission denied when saving to {self.data_file}: {e}")
            print("Ensure the file is not open in another program and try again.")
            raise
        
    def collect_and_process_data(self, brands, force_refresh=False):
        """Collect data from API or use stored data."""
        if force_refresh or self.items_df.empty:
            print("Fetching new data from API...")
            new_data = pd.DataFrame()
            for brand in brands:
                df = collect_poshmark_data(brand)
                if not df.empty:
                    processed_df = df[[
                        'title', 'brand', 'color', 'gender', 'type', 'price',
                        'original_price', 'size', 'condition', 'department',
                        'category', 'days_to_sell'
                    ]].dropna(how='all')
                    if not processed_df.empty:
                        for col in ['price', 'original_price', 'days_to_sell']:
                            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                        for col in processed_df.columns:
                            if col not in ['price', 'original_price', 'days_to_sell']:
                                processed_df[col] = processed_df[col].astype(str)
                        new_data = pd.concat([new_data, processed_df], ignore_index=True)
            
            if not new_data.empty:
                self.items_df = pd.concat([self.items_df, new_data], ignore_index=True)
                self.items_df.drop_duplicates(subset=['title'], keep='last', inplace=True)
                self.save_data()
            print(f"Collected {len(new_data)} new items. Total items: {len(self.items_df)}")
        else:
            print(f"Using stored data with {len(self.items_df)} items.")
        
    def preprocess_data(self):
        df_processed = self.items_df.copy()
        
        condition_map = {
            'nwt': 'New',
            'new with tags': 'New',
            'not_nwt': 'Used',
            'used': 'Used',
            'used - excellent': 'Used',
            'used - good': 'Used',
            'used - fair': 'Used',
            'ret': 'Used',
            'unknown': 'Unknown'
        }
        
        categorical_columns = ['brand', 'color', 'gender', 'type', 'size', 
                             'condition', 'department', 'category']
        for column in categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
            if column == 'condition':
                df_processed[column] = df_processed[column].astype(str).map(
                    lambda x: condition_map.get(x.lower(), 'Unknown')
                ).str.lower()
            else:
                df_processed[column] = df_processed[column].astype(str).str.lower()
            df_processed[column] = self.label_encoders[column].fit_transform(
                df_processed[column].fillna('unknown')
            )
            
        df_processed['price'] = pd.to_numeric(df_processed['price'], errors='coerce').fillna(0)
        df_processed['original_price'] = pd.to_numeric(df_processed['original_price'], errors='coerce').fillna(df_processed['price'])
        df_processed['days_to_sell'] = pd.to_numeric(df_processed['days_to_sell'], errors='coerce').fillna(-1)
        
        return df_processed

    def train_model(self):
        if len(self.items_df) < 10:
            return "Insufficient data to train model"
            
        processed_data = self.preprocess_data()
        
        features = ['brand', 'color', 'gender', 'type', 'price', 'original_price',
                   'size', 'condition', 'department', 'category']
        X = processed_data[features]
        
        valid_days = processed_data['days_to_sell'][processed_data['days_to_sell'] >= 0]
        if len(valid_days) == 0:
            return "No valid days_to_sell data available for training"
        self.median_days = valid_days.median()
        y = (processed_data['days_to_sell'].apply(lambda x: x < self.median_days if x >= 0 else False)).astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        return f"Model trained with accuracy: {self.model.score(X_test, y_test):.2f}"

    def predict_popularity(self, brand, color, gender, type, price, original_price,
                         size, condition, department, category):
        condition_map = {
            'nwt': 'New',
            'new with tags': 'New',
            'not_nwt': 'Used',
            'used': 'Used',
            'used - excellent': 'Used',
            'used - good': 'Used',
            'used - fair': 'Used',
            'ret': 'Used',
            'unknown': 'Unknown'
        }
        
        input_data = pd.DataFrame({
            'brand': [brand.lower()],
            'color': [color.lower()],
            'gender': [gender.lower()],
            'type': [type.lower()],
            'price': [float(price)],
            'original_price': [float(original_price)],
            'size': [size.lower()],
            'condition': [condition_map.get(condition.lower(), 'Unknown').lower()],
            'department': [department.lower()],
            'category': [category.lower()]
        })
        
        for column in self.label_encoders:
            input_data[column] = self.label_encoders[column].transform(input_data[column].astype(str))
            
        probability = self.model.predict_proba(input_data)[0][1]
        return f"Probability of selling in less than {self.median_days:.0f} days: {probability:.2%}"

    def analyze_trends(self):
        if len(self.items_df) == 0:
            return "No data available for analysis"
            
        trends = {}
        trends['top_brands'] = self.items_df['brand'].value_counts().nlargest(5).to_dict()
        trends['top_colors'] = self.items_df['color'].value_counts().nlargest(5).to_dict()
        trends['top_types'] = self.items_df['type'].value_counts().nlargest(5).to_dict()
        trends['avg_price_by_type'] = self.items_df.groupby('type')['price'].mean().to_dict()
        valid_days = self.items_df[self.items_df['days_to_sell'] >= 0]['days_to_sell']
        trends['avg_days_to_sell'] = valid_days.mean() if not valid_days.empty else float('nan')
        
        return trends

if __name__ == "__main__":
    analyzer = ResellMarketAnalyzer(data_file="resell_data.csv")
    
    brands = ['nike', 'converse', 'adidas']
    analyzer.collect_and_process_data(brands, force_refresh=False)  # Set to False to use stored data
    
    print(analyzer.train_model())
    
    trends = analyzer.analyze_trends()
    print("\nMarket Trends:")
    print(f"Top Brands: {trends['top_brands']}")
    print(f"Top Colors: {trends['top_colors']}")
    print(f"Top Types: {trends['top_types']}")
    print(f"Avg Price by Type: {trends['avg_price_by_type']}")
    print(f"Avg Days to Sell: {trends['avg_days_to_sell']:.1f}")
    
    prediction = analyzer.predict_popularity(
        brand="Nike",
        color="Black",
        gender="Male",
        type="Shoe",
        price=100,
        original_price=150,
        size="10",
        condition="Used - Excellent",
        department="Unknown",
        category="Shoes"
    )
    print(f"\nPrediction: {prediction}")