import pandas as pd

# Load the transactions data
transactions = pd.read_csv('data/transactions.csv')

# Initialize a dictionary to keep track of the last nonce for each sender
last_nonce = {}
results = []

def place_order_before(tx, index, next_tx):
    # Simulate the action of placing an order before the next transaction
    # Determine the profit or loss based on some condition
    # Example condition: If next transaction's value is higher, assume a profit; otherwise, a loss
    
    # Simple rule-based logic for profit/loss determination
    if next_tx is not None:
        if next_tx['value'] > tx['value']:
            assumed_result = 0.01  # Example profit
        else:
            assumed_result = -0.01  # Example loss
    else:
        assumed_result = 0.01  # Default to profit if no next transaction (end of dataset)
    
    return assumed_result, index

def backtest(transactions):
    global last_nonce
    for index in range(len(transactions) - 1):
        tx = transactions.iloc[index]
        next_tx = transactions.iloc[index + 1]
        
        sender = tx['from']
        nonce = tx['nonce']
        
        if sender in last_nonce and nonce == last_nonce[sender] + 1:
            print(f"Batch transaction detected from {sender}: {tx['hash']}")
            # Place an order before the next expected transaction
            result, idx = place_order_before(tx, index, next_tx)
            results.append((tx['hash'], result, 'Profitable' if result > 0 else 'Loss', idx))
        
        last_nonce[sender] = nonce

    return results

# Run the backtesting
results = backtest(transactions)

# Create a DataFrame to store the results
results_df = pd.DataFrame(results, columns=['Transaction Hash', 'Result', 'Classification', 'Index'])

# Save the results to a CSV file
results_df.to_csv('data/seq_nonce_backtesting_results.csv', index=False)


