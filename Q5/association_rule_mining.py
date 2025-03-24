import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Load the dataset
grocery_data_path = 'Groceries_dataset.csv'
grocery_data = pd.read_csv(grocery_data_path)

# Convert the 'Date' column to datetime format
grocery_data['Date'] = pd.to_datetime(grocery_data['Date'], format='%d-%m-%Y')

# Display the first few rows to check column names
print(grocery_data.head())
print(f"Transaction Info : {grocery_data.shape}")

# Generate individual itemsets
def individual_itemset(data):
    unique_items = set(data['itemDescription'])
    print(f"Total Unique Items : {len(unique_items)}")
    print(f"Unique Items : {unique_items}")
    return unique_items

unique_items = individual_itemset(grocery_data)

# Display the top 20 most frequent items
df = grocery_data["itemDescription"].value_counts()[:20].reset_index()
df.columns = ["Category", "Count"]
print(df)

# Create a unique transaction identifier
grocery_data["Single_transaction"] = grocery_data["Member_number"].astype(str) + "_" + grocery_data["Date"].astype(str)
print(grocery_data.head())

# Convert dataset into a list of transactions
transaction_data = grocery_data.groupby("Single_transaction")["itemDescription"].apply(list).tolist()
print(transaction_data[:5])

# Transaction Encoding
te = TransactionEncoder()
te_ary = te.fit(transaction_data).transform(transaction_data)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
print(df_encoded.head())
print(df_encoded.sum())

# Generating Itemsets
min_support = 0.01  # Minimum support threshold
frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
print(frequent_itemsets)

# Identifying Frequent Itemsets and Applying Apriori Algorithm
min_confidence = 0.01  # Minimum confidence threshold
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
print(rules)

# Evaluating Association Rules
filtered_data = rules[(rules['confidence'] > 0.1) & (rules['lift'] > 0.8)]
print(filtered_data)