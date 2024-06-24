# mempool_ai

## Project Steps:

1. **Listen to the Ethereum Mempool**
   Listens to the Ethereum mempool for pending transactions to gather real-time data.

2. **Label Sequential Nonces**
   Identifies and labels sequential nonces from the same sender in Ethereum transactions. This step is crucial for detecting potential batch transactions.

3. **Train a Transformer-Based Model**
   Uses a transformer-based architecture to predict the pseudo profit of subsequent trades based on the gathered and labeled data.

4. **Model Explainability and Feature Importance**
   Extracts and visualizes the model's explainability and feature importance.

   - **Feature Importance:**
     ![Feature Importance](assets/mempool_feat_importance.png)
   - **Model Explainability Report:**
     [Model Explainability Plots](ai/model_explainability_plots.ipynb)
