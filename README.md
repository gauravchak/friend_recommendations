# friend_recommendations
Recommending friends on a social network

## Versions

1. FriendingPredictionModelV1: an mvp implementation. sum pools the user sequence features and applies an MLP
2. FriendingPredictionModelV2: feature encoding is still minimal except introduces DCN and MMoE
3. FriendingPredictionModelV3: uses time scaling instead of sum pooling
4. FriendingPredictionModelV4: uses positional encoding instead of sum pooling
5. FriendingPredictionModelV5: uses time scaling, positional encoding and multiple self attention layers.

