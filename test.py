from SentimentNetwork import SentimentNetwork

g = open('reviews.txt','r') # What we know!
reviews = list(map(lambda x:x[:-1],g.readlines()))
g.close()

g = open('labels.txt','r') # What we WANT to know!
labels = list(map(lambda x:x[:-1].upper(),g.readlines()))
g.close()

print ("Initializing Neural Network...")
sn = SentimentNetwork(reviews[:-1000], labels[:-1000],
                    min_count=20, polarity_cutoff=0.8,
                    hidden_nodes = 20, learning_rate = 0.1)
print ("Training Network...")
sn.train(reviews[:-1000], labels[:-1000])
print ("\nTesting Network...")
sn.test(reviews[-1000:],labels[-1000:])
