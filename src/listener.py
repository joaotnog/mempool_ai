import os
import csv
import time
from web3 import Web3
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Alchemy WebSocket URL from environment variables
alchemy_ws_url = os.getenv('ALCHEMY_WS_URL')

# Connect to Alchemy's WebSocket
web3 = Web3(Web3.WebsocketProvider(alchemy_ws_url))

# Check connection
if web3.is_connected():
    print("Successfully connected to Alchemy WebSocket")
else:
    print("Failed to connect to Alchemy WebSocket")
    exit()

# Function to handle pending transactions
def handle_pending_transaction(event, csv_writer):
    try:
        tx = web3.eth.get_transaction(event)
        tx_data = {
            'hash': tx['hash'].hex(),
            'from': tx['from'],
            'to': tx['to'],
            'value': web3.from_wei(tx['value'], 'ether'),
            'gas': tx['gas'],
            'gas_price': web3.from_wei(tx['gasPrice'], 'gwei'),
            'nonce': tx['nonce'],
        }
        csv_writer.writerow(tx_data)
    except Exception as e:
        pass
        # print(f"Error processing transaction {event}: {e}")

def main():
    # Ensure the data directory exists
    os.makedirs('data', exist_ok=True)

    # Create and open the CSV file for writing
    with open('data/transactions.csv', mode='w', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=['hash', 'from', 'to', 'value', 'gas', 'gas_price', 'nonce'])
        csv_writer.writeheader()

        # Subscribe to pending transactions
        pending_filter = web3.eth.filter('pending')

        print("Listening for pending transactions...")

        # Keep the script running to listen for transactions
        try:
            while True:
                for event in pending_filter.get_new_entries():
                    handle_pending_transaction(event, csv_writer)
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopped the script")

if __name__ == "__main__":
    main()
