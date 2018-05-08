# Reference
# http://adventuresinmachinelearning.com/word2vec-tutorial-tensorflow/
# http://adventuresinmachinelearning.com/gensim-word2vec-tutorial/
# http://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/#.WuzEqohuaUk (about parameter of Word2Vec)


import pandas as pd
import numpy as np
import re
import os

import gensim

#%%
#Summary
os.chdir("C:\\Users\\snuist\\Desktop\\Git\\text-generator")

#import data
top100artist = pd.read_csv('./data/top100artist.csv', header=None) #raw data
data = top100artist.copy()
#set columns
data.columns = ['artist', 'song', 'link', 'text']
lyrics = data['text']

#data_cleansing
def data_cleansing(data):
    lyrics = str(data)
    lyrics = lyrics.lower()
    lyrics = lyrics.replace('\r', ' ')
    lyrics = lyrics.replace('\n', ' ')
    lyrics = lyrics.replace('.', ' EOS ') # . -> End Of Sentence
    lyrics = lyrics.replace('?', ' QM ') # ? -> Q (Question mark)
    lyrics = lyrics.replace('!', ' EM ') # ! -> EM (Exclamation mark)
    lyrics = lyrics.replace(',', ' COMMA ') # , -> Comma
    lyrics = lyrics.replace("'", '') # for he's -> hes, it's -> its, don't -> dont
    lyrics = re.sub("[^a-zA-Z]"," ", lyrics)
    return lyrics + "END" # End of one song 

#cleansing all lyrics and making word tokens list
words = [] #A list of all lyric tokens
for i in range(len(lyrics)):
    tokens = data_cleansing(lyrics[i]).split()
    words.extend(tokens)

#Building vocabulary and Training the word2vec model
model = gensim.models.Word2Vec(words,
                               iter = 10, #the number of iteration(epochs) over the corpus
                               size=300, #the size of the dense vector to represent each token. 
                               window=10, #the maximum distance between the target word and it's neighboring word.
                               min_count=10,
                               workers=10)

# save and reload the model
#model.save("word2vec_model")
#model = gensim.models.Word2Vec.load("word2vec_model")


#%%
#


#Building vocabulary and Training the word2vec model
model = gensim.models.Word2Vec(words,
                               iter = 10, #the number of iteration(epochs) over the corpus
                               size=300, #the size of the dense vector to represent each token. 
                               window=10, #the maximum distance between the target word and it's neighboring word.
                               min_count=10,
                               workers=10)


# <pre>
# Parameter
# -size: the size of dense vector to represent each token. If we have limited data, then size should be a smaller value.
# -window: the maximum distance between the target word and it's neighboring word. If our data is not sparse, then the size of window should not matter too much, as long as it's not overly narrow or broad.
# -min_count: minimum frequeny count of words. The model would ignore words that do not satisfy the min_count. Unless our dataset is really tiny, this does not really affect the model in terms of our final result.
# -workers: how many threads to use behind the scenes?


#%%
vocabulary_size = 50000

def build_dataset(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
#del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
#%%
data_index = 0

# Step 3: skip-gram model을 위한 트레이닝 데이터(batch)를 생성하기 위한 함수.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
      '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
#%%
# Step 4: skip-gram model 만들고 학습시킨다.

batch_size = 128
embedding_size = 300  # embedding vector의 크기.
skip_window = 5       # 윈도우 크기 : 왼쪽과 오른쪽으로 얼마나 많은 단어를 고려할지를 결정.
num_skips = 2         # 레이블(label)을 생성하기 위해 인풋을 얼마나 많이 재사용 할 것인지를 결정.

# sample에 대한 validation set은 원래 랜덤하게 선택해야한다. 하지만 여기서는 validation samples을 
# 가장 자주 생성되고 낮은 숫자의 ID를 가진 단어로 제한한다.
valid_size = 16     # validation 사이즈.
valid_window = 100  # 분포의 앞부분(head of the distribution)에서만 validation sample을 선택한다.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # sample에 대한 negative examples의 개수.

graph = tf.Graph()

with graph.as_default():

  # 트레이닝을 위한 인풋 데이터들
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # embedding vectors 행렬을 랜덤값으로 초기화
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    # 행렬에 트레이닝 데이터를 지정
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # NCE loss를 위한 변수들을 선언
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # batch의 average NCE loss를 계산한다.
  # tf.nce_loss 함수는 loss를 평가(evaluate)할 때마다 negative labels을 가진 새로운 샘플을 자동적으로 생성한다.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  # SGD optimizer를 생성한다.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # minibatch examples과 모든 embeddings에 대해 cosine similarity를 계산한다.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)
#%%
# Step 5: 트레이닝을 시작한다.
num_steps = 100001

with tf.Session(graph=graph) as session:
  # 트레이닝을 시작하기 전에 모든 변수들을 초기화한다.
  tf.initialize_all_variables().run()
  print("Initialized")

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

    # optimizer op을 평가(evaluating)하면서 한 스텝 업데이트를 진행한다.
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # 평균 손실(average loss)은 지난 2000 배치의 손실(loss)로부터 측정된다.
      print("Average loss at step ", step, ": ", average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8 # nearest neighbors의 개수
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        log_str = "Nearest to %s:" % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = "%s %s," % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()
#%%
# Step 6: embeddings을 시각화한다.

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  #in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels)

except ImportError:
  print("Please install sklearn and matplotlib to visualize embeddings.")