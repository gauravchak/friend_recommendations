# Friend Recommendations

This repository contains various versions of models aimed at predicting friending interactions on a social network. The models progress from basic implementations to more complex architectures that leverage advanced techniques such as feature interactions, time-based attention mechanisms, and multi-task learning frameworks. Each version builds upon the previous one, improving both feature encoding and architectural design.

## Model Versions

### **FriendingPredictionModelV1**

The first version is a basic MVP implementation. It uses **mean pooling** to aggregate user interaction sequences (e.g., friending history) into fixed-sized feature vectors, which are then passed through a simple **Multi-Layer Perceptron (MLP)** to make predictions. This model serves as a baseline, providing essential functionality without complex feature interactions.

**Key Features:**

- Mean pooling to summarize user interaction sequences.
- MLP for predicting friending-related probabilities (e.g., friend requests, acceptance, messaging).

---

### **FriendingPredictionModelV2**

In V2, the model introduces the computation of **pairwise interactions** between features from both the viewer and the target user. These interactions allow the model to capture more granular relationships between users by considering not only their individual features but also how these features interact with one another.

**Key Features:**
- Pairwise feature interactions for higher-order relationships.
- The resulting interaction features are concatenated with the original features and passed through the MLP.

---

### **FriendingPredictionModelV3**

V3 brings in more advanced techniques with the introduction of **Deep & Cross Networks (DCN)** and **Multi-gate Mixture-of-Experts (MMoE)**. The model improves its ability to handle feature interactions and supports multi-task learning across several friending-related tasks.

**Key Features:**

- **Deep & Cross Network (DCN):** A series of cross layers are used to model feature interactions in a non-linear fashion. Each cross layer computes interactions between different feature sets.
- **MMoE (Multi-gate Mixture-of-Experts):** This multi-task learning framework uses shared expert networks and task-specific gating mechanisms to ensure tasks share useful information while still having task-specific outputs.
- **Layer Normalization:** Applied after the DCN layers to stabilize training and improve convergence.

---

### **FriendingPredictionModelV4**

In V4, we introduce the concept of **timegap-based weighted pooling**. Instead of simply mean pooling user interaction sequences, the model now uses a small MLP to generate attention weights based on the time gaps between user interactions. This allows the model to weigh more recent interactions higher than older ones, making the sequence aggregation more dynamic.

**Key Features:**

- **TimegapWeightedSum:** This module computes attention scores from time gaps between user interactions and applies them to compute a weighted sum of the interaction embeddings. This ensures that the model pays more attention to interactions that happened more recently.
- **DCN and MMoE Layers:** The weighted interaction embeddings are passed through the existing DCN and MMoE layers to capture feature interactions and support multi-task learning.

---

### **FriendingPredictionModelV5**

V5 is the most complex and powerful version of the model. It introduces **time scaling**, **positional encoding**, and **self-attention mechanisms** to better capture both the sequence and temporal dynamics of user interactions. This version is particularly useful for capturing dependencies between interactions that occur at different times.

**Key Features:**

1. **Positional Encoding:**
   - To account for the order of user interactions, the model uses positional encoding similar to Transformer architectures. This helps encode the temporal position of interactions in the sequence.

2. **Complex Time Scaling:**
   - A time-scaling module processes the time gaps between interactions and applies a non-linear transformation to these gaps. The scaled time gaps are then used to modulate the attention mechanism.

3. **Time-Scaled Attention:**
   - The attention mechanism in V5 is extended to incorporate time scaling. The attention scores between different interaction embeddings are adjusted using time scaling, giving more weight to interactions that occur closer in time.

4. **Self-Attention Layers:**
   - Multiple self-attention layers allow the model to capture complex dependencies between different user interactions, using both content-based attention and time-scaled attention.

5. **Deep & Cross Network (DCN) and MMoE:**
   - After the time-scaled attention layers, the resulting features are processed through the same DCN and MMoE layers as in V3 and V4, combining the benefits of feature interaction modeling and multi-task learning.

**Why V5?**
V5 is designed to capture not just the static relationships between users but also how their interactions evolve over time. By incorporating time gaps, positional encoding, and attention mechanisms, this version can model complex temporal dynamics that are often crucial in predicting friending behaviors.

---

## Conclusion

Each version of the **FriendingPredictionModel** builds upon the previous one, introducing increasingly sophisticated techniques for modeling user interactions on social networks. From basic MLP-based predictions to advanced models with self-attention, time scaling, and multi-task learning, this repository provides a comprehensive exploration of state-of-the-art methods for social recommendation systems.

Feel free to explore the models, tweak the parameters, and experiment with your own datasets to further refine the predictions.
