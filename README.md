# Sentiment Classification Network
Neural network for sentiment classification on natural language

This network classifies text movie reviews between positive and negative. By running the test file:

```
python test.py
```
you would probably get something like this:

```
Initializing Neural Network...
Training Network...
Progress:99.9% Speed(reviews/sec):5963. #Correct:20593 #Trained:24000 Training Accuracy:85.8%
Testing Network...
Progress:99.9% Speed(reviews/sec):7513. #Correct:808 #Tested:1000 Testing Accuracy:80.8%%
```
However, you can place with the parameters to get more interesting results:

 - min_count: filters the unfrequent words that appeared in training data set
 - polarity_cutoff: filters neutral words. the lower this parameter is, the more the network accepts neutral words (such as: 'the', 'and', 'a', 'I') into consideration
 - hidden_nodes: defines the number of hidden layer units. default to be 10.

I also added an implementation of the network using tensorflow. to run the code:

```
python TFN.py
```
