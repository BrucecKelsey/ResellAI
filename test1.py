import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from scrape import collect_poshmark_data
# Uncomment to suppress FutureWarning
# import warnings
# warnings.filterwarnings('ignore', category=FutureWarning)

class ResellMarketAnalyzer:
    def __init__(self):
        self.items_df = pd.DataFrame(columns=[
            'title', 'brand', 'color', 'gender', 'type', 'price',
            'original_price', 'size', 'condition', 'department',
            'category', 'days_to_sell'
        ], dtype=object)
        self.label_encoders = {}
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def collect_and_process_data(self, brands):
        for brand in brands:
            df = collect_poshmark_data(brand)
            if not df.empty:
                processed_df = df[[
                    'title', 'brand', 'color', 'gender', 'type', 'price',
                    'original_price', 'size', 'condition', 'department',
                    'category', 'days_to_sell'
                ]].dropna(how='all')
                if not processed_df.empty:
                    for col in processed_df.columns:
                        processed_df[col] = processed_df[col].astype(object)
                    self.items_df = pd.concat([self.items_df, processed_df], ignore_index=True)
        
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
            
        df_processed['price'] = df_processed['price'].astype(float).fillna(0)
        df_processed['original_price'] = df_processed['original_price'].astype(float).fillna(df_processed['price'])
        df_processed['days_to_sell'] = df_processed['days_to_sell'].astype(float).fillna(-1)
        
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
        median_days = valid_days.median()
        y = (processed_data['days_to_sell'].apply(lambda x: x < median_days if x >= 0 else False)).astype(int)
        
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
        return f"Probability of quick sale: {probability:.2%}"

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
    analyzer = ResellMarketAnalyzer()
    
    brands = ['nike', 'converse', 'adidas']
    analyzer.collect_and_process_data(brands)
    
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