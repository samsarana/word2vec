# Word2vec-research
Contains code used to investigate the application of Word2vec style algorithms to model a train of thought and my final report.

The report explains my research aims and outcomes.

The python program 'Train word2vec on statement_logic.py' produced my final result which was a dictionary mapping informal logic statements to high dimensional vectors, where statements that should follow from each other in good logical reasoning and trained to be embedded closely to each other (i.e. with high cosine similarity).
That dictionary can be loaded using numpy.load('models_dict2.npy').item()

As for the other python programs:
 - the word2vec_modified programs are my modified implementations of word2vec, using for training (modifications explained in the report)
 - statement_logic.py was used to produce the informal logic manipulations for training, which can be viewed in the .zip file
 - the rest are modules (or modifications to modules), used by statement_logic.py to generate the trains of reasoning
