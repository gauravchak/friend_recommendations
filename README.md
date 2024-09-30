# friend_recommendations
Recommending friends on a social network

## Versions

1. FriendingPredictionModelV1: an mvp implementation. This mean pools the user sequence features and applies an MLP to get task predictions.
2. FriendingPredictionModelV2: feature encoding is still minimal except for each of the 4 features we compute pairwise interactions.
3. FriendingPredictionModelV3: feature encoding is still minimal except introduces DCN and MMoE
4. FriendingPredictionModelV4: uses time scaling instead of mean pooling
5. FriendingPredictionModelV5: uses time scaling, positional encoding and multiple self attention layers.

