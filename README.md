# friend_recommendations
Recommending friends on a social network

In various versions we will be iterating through increasingly complex architectures for predicting friending interactions, each version building on the previous one in terms of feature encoding and model architecture. Here's a breakdown of the key enhancements as you move from version to version:

## FriendingPredictionModelV1

Basic MVP implementation using a mean pooling of user sequence features and a simple MLP (Multi-Layer Perceptron) for predictions.
The features from both the viewer and target users are concatenated and passed through the MLP.

## FriendingPredictionModelV2:

Introduces the computation of pairwise interactions between features of the viewer and target. These interactions capture more granular relationships between the feature dimensions.
After calculating pairwise feature interactions, the model concatenates them with the original features and feeds them into the MLP.

1. FriendingPredictionModelV3: feature encoding is still minimal except introduces DCN and MMoE
2. FriendingPredictionModelV4: uses time scaling instead of mean pooling
3. FriendingPredictionModelV5: uses time scaling, positional encoding and multiple self attention layers.

