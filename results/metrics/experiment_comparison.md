# Experiment Comparison

| dataset          | model        |   overall_accuracy |   average_accuracy |   kappa |   train_pixels |   test_pixels |
|:-----------------|:-------------|-------------------:|-------------------:|--------:|---------------:|--------------:|
| Pavia University | SVM + PCA    |             0.9542 |             0.9402 |  0.9392 |          29943 |         12833 |
| Pavia University | RandomForest |             0.9262 |             0.8978 |  0.9011 |          29943 |         12833 |
| Salinas          | RandomForest |             0.9512 |             0.9747 |  0.9456 |          37890 |         16239 |
| Salinas          | SVM + PCA    |             0.9394 |             0.9730 |  0.9324 |          37890 |         16239 |

## Best Model Per Dataset

- Pavia University: SVM + PCA (OA=0.9542, AA=0.9402, Kappa=0.9392)
- Salinas: RandomForest (OA=0.9512, AA=0.9747, Kappa=0.9456)
