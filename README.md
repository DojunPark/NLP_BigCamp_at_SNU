# SNU: Boot Camp for NLP
The 7th data sience boot camp BIG CAMP with big data institute of Seoul National University <br/>
From Oct. 22. 2019 to Oct. 25. 2019 in Gaepo Digital Innovation Centre<br/>

# Summaries of lectures (English)

## Day 1

In the field of natural language processing, an RNN module that can handle sequences has been used for some time. <br/>
The core of natural language processing is embedding starting at the first layer <br/>
Good embedding is a very important part <br/>
Also, natural language processing has a long history ex) Google's machine translator (has made great progress since the transformer) <br/>
He received a lot of investment since World War II. <br/>
The history of natural language processing is the history of machine translation. <br/>
Initially, grammar-based-> statistical-based-> artificial neural network based-> transformer-based <br/>
The fields of natural language processing include voice assistant (Siri), chatbot (Q & A), and summary machine. <br/>

Computers don't understand our speech, so we have to make it understandable. <br/>
-Delivered in a computer-readable representation <br/>
Must understand how human language is made <br/>
-Speech (acoustic transmission)-> Phoneme analysis-> Syntax analysis-> Semantic analysis-> Context inference <br/>

In many cases, the sum of each element can be different from the whole <br/>
ex) Why is it so hot here? -> It may imply that the air conditioner is turned on. <br/>

### Can the computer understand the language?
Part-of-speech analyzers and syntax analyzers have been developed, but can computers understand them? <br/>
Difficult to understand as we want <br/>
But another approach is in progress <br/>
Human language is very ambiguity (ambiguity) <br/>

### Create a pipeline
Do everything one by one and connect them all <br/>
1. Sentence segmentation <br/>
 -> Not always divided by period (decimal point, abbreviation) <br/>
2. Divide by word (word tokenization) <br/>
 -> Not always divided by spaces (New York) <br/>
3. Predicting POS <br/>
 -> Using the morpheme analyzer to designate parts of speech <br/>
4. Lemmatization <br/>
 -> The form has changed, but the same meaning is restored to the original word <br/>
    There is no circular restoration module in Korean <br/>
5. Identifying stop words <br/>
6. Dependency analysis of sentence structure (dependency parsing) <br/>
 -> Stanford parser, Google parser, etc. in English <br/>
    None in Korean <br/>
    In the case of English, the word order is fixed and the morpheme is very complicated (if you've eaten it) <br/>
    But Korean is relatively free in word order. <br/>
6-2. Finding noun phrases <br/>
7. Named Entity Recognition <br/>
8. Coreference Resolution <br/>
 -> In English, it is very important because pronouns are actively used. <br/>
    Korean is relatively insignificant (he doesn't write her, etc.) <br/>
Dividing everything into smaller ones and then integrating them into the pipeline <br/>

### Useful natural language processing modules
-spaCy
-textacy

### Extract semantics using classification model
Most problems in natural language processing are classification problems. <br/>
Put review text and make a classification model-> If you put text, you get a rating. <br/>
- How does this work?
There are many reviews that generate a lot of reasons why the classification model works well, and it works by finding out the features hidden in the text. <br/>
- Makes sense of meaning very fast
Depending on the pipeline, it takes a long time to proceed, but the classification model is calculated by linear regression or logistic regression, so you can get answers very quickly <br/>
- Requires large training data
Small data set makes learning features difficult <br/>
The larger the data set, the higher the accuracy <br/>
- Review analyzer, spam classifier, etc ...
- Everything can be regarded as a classification problem, such as understanding meaning and machine translation
- Embedding also can analyze the meaning of words

TPU performs parallel calculation better <br/>
Borrow a small amount from Google <br/>


## Day 2

Learning semantic reasoning of words from last time by classification <br/>
Association concept-RNN, LSTM, encoder and decoder, Attention <br/>

### RNN
Characteristics of language data-sequentiality <br/>
RNN Network <br/>
Data linked in sequence-sequential data <br/>
Ex) Sound, language ... <br/>
Hidden layers are rotated to connect sequential information. <br/>
RNN cannot remember old data due to short term memory <br/>

### LSTM
Model to solve the vanishing gradient of RNN <br/>
RNN had only one hidden layer, whereas LSTM has an added long-term memory layer called a cell <br/>
Decide what to forget and keep <br/>
Add functions such as sigmoid and tangent hyperboli to each gate <br/>
Forget, input, output… Have a gate <br/>
1. Using the sigmoid value, make the forgotten by forgetting it. <br/>
2. One through sigmoid, one through tangent hyperboli (new features added) <br/>
   The value to be forgotten and the value to be added are transferred to the cell <br/>
3. The output gate determines what happens next. <br/>

### GRU
Simplified LSTM with reset gate and output gate <br/>
Fast and excellent performance. <br/>
A model first devised by Prof. Kyung-Hyun Cho at New York University

### Seq2seq
While the existing RNN obtains the result value as the last value, the encoder decoder model of seq2seq converts the input sentence into one vector with the encoder and converts the vector into the target language with the decoder <br/>
The vector generated by the encoder from the input seq is called a context vector <br/>
However, the longer the sentence is, the longer it is expressed in one limited vector. <br/>
RNN gets stuck in the basic problem <br/>
The attention model comes out to solve this encoder decoder problem <br/>
In LSTM, it is important to pay attention <br/> to calculate where to pay attention as a score, such as dividing keep and forget.
Decide where to focus on encoding <br/>
Let's use all encoding information in decoding-> Create attention layer between encoder and decoder <br/>
Provides hidden state of all encoders <br/>
The encoder-decoder model is a model that is frequently used in translation models. <br/>
When performing machine translation, it is important to align how the source and target are connected (attention) <br/>

### attention
Establish relationship between which input value is connected to which output value in each hidden state <br/>
Decoder's hidden state and encoder's hidden state are reflected and reflected <br/>
Dot-product is to find out how similar the two values ​​are. <br/>
Converting each score to a probability value takes a softmax function <br/>
In the next Attention layer, the probability value is multiplied by the original value <br/>
Attach the most important value of the attention to the decoder <br/>
Dot-product value is similarity <br/>

### Example of Attention
Bahdanau-Pioneer of Attention Model <br/>
Luong-Stack two encoder layers and proceed <br/>
Go forward, go back and go <br/>
How many layers of encoding are stacked varies from person to person. <br/>
GNMT-Decoding the attention value through 8 LSTMs in 2016 <br/>
In the last output layer, it is passed as output value without going through ResNet (residual net) <br/>
I go through several hidden layers, but sometimes it's better to get the output without going <br/>
Value from 8 <br/>

### summary
Don't make a single vector, but use the value of each state <br/>

### Questions
What is the difference between remembering positional information in multihead attention and remembering positional information in positional encoding? <br/>
What is Value? <br/>


## Day 3

### Transformer
Convert the value of each word into a vector and put it in self-attention <br/>
Add attention scores from each word to the feedforward <br/>
Check everything and find out which word has the strongest attention and which word <br/>

Query score to get attention score with three values: query, key and value <br/>
The value obtained by comparing the query with the key is value <br/>
Similarity is calculated by dot product in a vector (dot-product) <br/>
Takes Softmax and converts it to probability <br/>
The value of self and self-introduction is the highest. <br/>

Query-words to be compared (target language) <br/>
Key-Words to be compared ㅏ (source language) <br/>
Value-(source language) <br/>

Multiplying the query with the key is finding the similarity <br/>
Not to be affected by the length of K <br/>

Originally from machine translation <br/>
For input French and output English, query in English and key and value in French <br/>

### positional encoding
-To put the positive information <br/>

Values ​​other than yourself are prevented from being attracted. <br/>
If wrong, the weight is adjusted <br/>

### Questions
How does concating by stacking 8 layers affect performance? <br/>
-> Every weight is different <br/>

Not sequential, and can be pararellized <br/>
It can be visualized for each layer so that it can be tracked (with a tool called Burt Beads) <br/>

## Day 4

### BERT

Actually, BERT is used rather than the transformer. <br/>
BERT is a pre-trained model <br/>
There are many other pre-learning models since BERT <br/>

Many innovative models of NLP appeared in 2018 last year. <br/>
-ELMo, BERT, GPT, ULM-FiT… <br/>
-Changed the paradigm of sequential model in BERT <br/>
-Pre-training can be used everywhere, and all other tasks are performed with fine-tuning. <br/>
(Part that was predicted to be difficult in the past) <br/>
-1. Trained model that teaches supervised through semi-supervised learning <br/>
2. Perform specific tasks through supervised learning <br/>
-Meaning extract is not easy <br/>
Semantic extraction from text is a classification task (positive, negative, neutral) <br/>
-Most of the tasks of NLP are classified <br/>
-BERT performance surpasses other models in most tasks <br/>

### Sentence Classification
-Pre-learning by receiving input-> fine-tuning <br/>

There are BERT base and BERT large <br/>
-Base is the same size as GPT
-BERT is a learned transformer encoder stack
-BERT is made only with Transformer encoder (no translation task is required, so no decoder is required)
-BERT base has 12 encoders and large has 24
-Hidden layer increased to 768 and 1024, multi-head attention increased to 12 and 16
-[CLS] token is added at the beginning of the sentence
-> To extract or classify word-specific features
-If you know how to use BERT, you can easily perform all tasks

### Features of BERT
-Only Transformer Encoder is used
-Perform feature extraction and classification tasks with BERT's pre-learning model
-In the case of RNN and CNN, you had to learn it yourself, but since BERT is already pre-trained, you can perform tasks only by fine-tuning.
-Transformer is not used unless you create your own translator

### Embedding
-Previously, one-hot encoding
-One-hot encoding cannot understand the semantic relationship between words
-Grasping the meaning of words through surrounding words is word embedding
Word to vector-> word2vec
-Word embedding contributes greatly to text processing
-Learning method is learning in relation to surrounding words
-Expressed as 100-dimensional vector, 10000-dimensional vector, etc.
The larger the dimension of the vector, the finer the meaning can be expressed.

### ELMo
-Solve the problem of synonyms
-The existing word embedding could not solve the problem of synonyms.
-Learning the context of words to learn different vector values ​​for each word
-Coming in 2018
-Good performance by guessing context with 3 layers
-Predict words to come later with language modeling
-Write LSTM layer, learn in both forward and backward directions and concat the two layers
-Multiply each layer by its weight
-Embedding only when you put a sentence
-Even with the same stick, it has a different vector value (because it was learned with different sentences)
-Extraction of a vector of a specific word from the embedding value (feature extraction)
-In case of ELMo, if the layer goes up, it learns the context vector of the word well.

ULM-FiT also pre-learns <br/>
Transformer <br/>
OpenAI Transformer <br/>
-Learning with only the decoder before BERT comes out <br/>
-Reference value x <br/> because no encoder is inserted

### BERT
-Refer to the transformer's encoder value
-The bidirectionality of ELMo concatenated two directions
-BERT learns bidirectional at one point at the same time (available by masking)
-Learning value is contained in [CLS]
-15% chance to replace [mask] with other words or existing words
-Improves performance by learning these words as words before and after
-By learning this, you can learn in both directions
-[SEP] token is put at the end of the first and second sentences
-1. Train the [mask] value
-2. Train whether the first sentence and the second sentence are consecutive sentences
-BERT embedding can understand the meaning of words with different meanings depending on the context like ELMo
-Is the learning value of the last layer the best?
Since each layer has a different learning value, the sum of all layers has the best learning value
In other tasks, the sum of the second to last layers is the best learning value.
-Feature-extraction is also possible.
-Object name recognition

### BERT FEATURES
-Wordpiece
-Tokenizing to remove unknown tokens
-Performance depends on how you cut the wordpiece
-3 tokens are combined to get the value

### Fine-tuning
-Proceed with a real pre-training model
-cls is binary classification
-When classifying by multi-class, softmax is applied to the last hidden layer to be multi-class.
-Shows the best performance for all tasks

### XLNet
-Autoregressive LM?
Predict with two words before or after
Predict word in context word (GPT…)
Great performance for typical NLP tasks
However, the characteristic of not being able to learn simultaneously
-Burt is an autoencoder
Predict the word [mask] from before and after
The disadvantage of Burt is pre-learning with a random [mask].
For example, in the case of a baking crisis, two words are dependent but masked independently.
-So use permutation
If there is a certain position, permute each other, and the number of possible cases will be as many as the total number of tokens.
Learning with tokens of t-1 of t
-Great idea
This is the case for Korean language surveys
However, since all cases need to be calculated as factorial, the calculation amount is very high

### ETRI's Korbert
-Morphological analyzer
-Classifier and feature-extraction can be used according to purpose
SKT's KoBERT <br/>
Seoul National University is also making it-in a different way (but difficult ;;;) <br/>

### Fine-tuning Practice
Take the pretrained weight of the hugging face and do the crash

## Day 5

If you do morphological analysis, <br/>
I should have eaten ... Analysis of meaningless relationship between morphemes becomes <br/>
Korean has other issues related to embedding <br/>
The basic idea of Word2vec is skip-gram (prediction of surrounding words as a central word) / cbow <br/>

Word2vec is a token unit <br/>
Fasttext is divided into subword units <br/>
using a lot of glove in most English <br/>

Brings pre-trained word embeddings <br/>
