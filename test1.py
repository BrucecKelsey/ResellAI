import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from scrape import collect_poshmark_data  # Import from separate file

class ResellMarketAnalyzer:
    def __init__(self):
        self.items_df = pd.DataFrame(columns=[
            'title', 'brand', 'color', 'gender', 'type', 'price',
            'original_price', 'size', 'condition', 'department',
            'category', 'days_to_sell'
        ])
        self.label_encoders = {}
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def collect_and_process_data(self, brands):
        """Collect and process data for multiple brands"""
        for brand in brands:
            df = collect_poshmark_data(brand)
            if not df.empty:
                processed_df = df[[
                    'title', 'brand', 'color', 'gender', 'type', 'price',
                    'original_price', 'size', 'condition', 'department',
                    'category', 'days_to_sell'
                ]]
                self.items_df = pd.concat([self.items_df, processed_df], ignore_index=True)
        
    def preprocess_data(self):
        """Prepare data for analysis"""
        df_processed = self.items_df.copy()
        
        # Encode categorical variables
        categorical_columns = ['brand', 'color', 'gender', 'type', 'size', 
                             'condition', 'department', 'category']
        for column in categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
            df_processed[column] = self.label_encoders[column].fit_transform(
                df_processed[column].fillna('Unknown')
            )
            
        # Handle numerical columns
        df_processed['price'] = df_processed['price'].fillna(0)
        df_processed['original_price'] = df_processed['original_price'].fillna(df_processed['price'])
        df_processed['days_to_sell'] = df_processed['days_to_sell'].fillna(-1)
        
        return df_processed

    def train_model(self):
        """Train the AI model"""
        if len(self.items_df) < 10:  # Increased minimum data requirement
            return "Insufficient data to train model"
            
        processed_data = self.preprocess_data()
        
        # Features for training
        features = ['brand', 'color', 'gender', 'type', 'price', 'original_price',
                   'size', 'condition', 'department', 'category']
        X = processed_data[features]
        
        # Target: fast-selling items (sold in less than median time)
        median_days = processed_data['days_to_sell'][processed_data['days_to_sell'] >= 0].median()
        y = (processed_data['days_to_sell'].apply(lambda x: x < median_days if x >= 0 else False)).astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        return f"Model trained with accuracy: {self.model.score(X_test, y_test):.2f}"

    def predict_popularity(self, brand, color, gender, type, price, original_price,
                         size, condition, department, category):
        """Predict if an item will sell quickly"""
        input_data = pd.DataFrame({
            'brand': [brand.lower()],
            'color': [color.lower()],
            'gender': [gender.lower()],
            'type': [type.lower()],
            'price': [float(price)],
            'original_price': [float(original_price)],
            'size': [size.lower()],
            'condition': [condition.lower()],
            'department': [department.lower()],
            'category': [category.lower()]
        })
        
        for column in self.label_encoders:
            input_data[column] = self.label_encoders[column].transform(input_data[column])
            
        probability = self.model.predict_proba(input_data)[0][1]
        return f"Probability of quick sale: {probability:.2%}"

    def analyze_trends(self):
        """Analyze market trends"""
        if len(self.items_df) == 0:
            return "No data available for analysis"
            
        trends = {}
        trends['top_brands'] = self.items_df['brand'].value_counts().nlargest(5).to_dict()
        trends['top_colors'] = self.items_df['color'].value_counts().nlargest(5).to_dict()
        trends['top_types'] = self.items_df['type'].value_counts().nlargest(5).to_dict()
        trends['avg_price_by_type'] = self.items_df.groupby('type')['price'].mean().to_dict()
        trends['avg_days_to_sell'] = self.items_df[self.items_df['days_to_sell'] >= 0]['days_to_sell'].mean()
        
        return trends

# Example usage
if __name__ == "__main__":
    analyzer = ResellMarketAnalyzer()
    
    # Collect data for multiple brands
    brands = ['nike', 'converse', 'adidas']
    analyzer.collect_and_process_data(brands)
    
    # Train model
    print(analyzer.train_model())
    
    # Analyze trends
    trends = analyzer.analyze_trends()
    print("\nMarket Trends:")
    print(f"Top Brands: {trends['top_brands']}")
    print(f"Top Colors: {trends['top_colors']}")
    print(f"Top Types: {trends['top_types']}")
    print(f"Avg Price by Type: {trends['avg_price_by_type']}")
    print(f"Avg Days to Sell: {trends['avg_days_to_sell']:.1f}")
    
    # Make a prediction
    prediction = analyzer.predict_popularity(
        brand="Nike",
        color="Black",
        gender="Male",
        type="Shoe",
        price=100,
        original_price=150,
        size="10",
        condition="Used - Excellent",
        department="Men",
        category="Sneakers"
    )
    print(f"\nPrediction: {prediction}")