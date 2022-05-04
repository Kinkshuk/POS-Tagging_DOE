# Importing the required libraries 
import nltk
import numpy as np
import itertools
# Downloading and importing the brown corpus
x = open("trump.txt","r")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
tokens = nltk.word_tokenize(x.read())
final_tags = []
for i in tokens:
    final_tags.extend(nltk.pos_tag(nltk.word_tokenize(i)))
print(final_tags[:2])
sent_tag = [final_tags]
print(sent_tag)
mod_sent_tag=[]
for s in sent_tag:
  s.insert(0,('##','##'))
  s.append(('&&','&&'))
  mod_sent_tag.append(s)
#Splitting the data for train and test
split_num = int(len(mod_sent_tag)*0.9)
train_data = mod_sent_tag[0:split_num]
test_data = mod_sent_tag[split_num:]
#Creating a dictionary whose keys are tags and values contain words which were assigned the correspoding tag
# ex:- 'TAG':{word1: count(word1,'TAG')}
train_word_tag = {}
for s in train_data:
  for (w,t) in s:
    w=w.lower()
    try:
      try:
        train_word_tag[t][w]+=1
      except:
        train_word_tag[t][w]=1
    except:
      train_word_tag[t]={w:1}
#Calculating the emission probabilities using train_word_tag
train_emission_prob={}
for k in train_word_tag.keys():
  train_emission_prob[k]={}
  count = sum(train_word_tag[k].values())
  for k2 in train_word_tag[k].keys():
    train_emission_prob[k][k2]=train_word_tag[k][k2]/count
#Estimating the bigram of tags to be used for transition probability
bigram_tag_data = {}
for s in train_data:
  bi=list(nltk.bigrams(s))
  for b1,b2 in bi:
    try:
      try:
        bigram_tag_data[b1[1]][b2[1]]+=1
      except:
        bigram_tag_data[b1[1]][b2[1]]=1
    except:
      bigram_tag_data[b1[1]]={b2[1]:1}
#Calculating the probabilities of tag bigrams for transition probability  
bigram_tag_prob={}
for k in bigram_tag_data.keys():
  bigram_tag_prob[k]={}
  count=sum(bigram_tag_data[k].values())
  for k2 in bigram_tag_data[k].keys():
    bigram_tag_prob[k][k2]=bigram_tag_data[k][k2]/count
#Calculating the possible tags for each word
#Note: Here we have used the whole data(Train+Test)
#Reason: There may be some words which are not present in train data but are present in test data 
tags_of_tokens = {}
count=0
for s in train_data:
  for (w,t) in s:
    w=w.lower()
    try:
      if t not in tags_of_tokens[w]:
        tags_of_tokens[w].append(t)
    except:
      l = []
      l.append(t)
      tags_of_tokens[w] = l
        
for s in test_data:
  for (w,t) in s:
    w=w.lower()
    try:
      if t not in tags_of_tokens[w]:
        tags_of_tokens[w].append(t)
    except:
      l = []
      l.append(t)
      tags_of_tokens[w] = l
#Dividing the test data into test words and test tags
test_words=[]
test_tags=[]
for s in test_data:
  temp_word=[]
  temp_tag=[]
  for (w,t) in s:
    temp_word.append(w.lower())
    temp_tag.append(t)
  test_words.append(temp_word)
  test_tags.append(temp_tag)
#Executing the Viterbi Algorithm
predicted_tags = []                #intializing the predicted tags
for x in range(len(test_words)):   # for each tokenized sentence in the test data
  s = test_words[x]
  #storing_values is a dictionary which stores the required values
  #ex: storing_values = {step_no.:{state1:[previous_best_state,value_of_the_state]}}                
  storing_values = {}              
  for q in range(len(s)):
    step = s[q]
    #for the starting word of the sentence
    if q == 1:                
      storing_values[q] = {}
      tags = tags_of_tokens[step]
      for t in tags:
        #this is applied since we do not know whether the word in the test data is present in train data or not
        try:
          storing_values[q][t] = ['##',bigram_tag_prob['##'][t]*train_emission_prob[t][step]]
        #if word is not present in the train data but present in test data we assign a very low probability of 0.0001
        except:
          storing_values[q][t] = ['##',0.0001]#*train_emission_prob[t][step]]
    
    #if the word is not at the start of the sentence
    if q>1:
      storing_values[q] = {}
      previous_states = list(storing_values[q-1].keys())   # loading the previous states
      current_states  = tags_of_tokens[step]               # loading the current states
      #calculation of the best previous state for each current state and then storing
      #it in storing_values
      for t in current_states:                             
        temp = []
        for pt in previous_states:                         
          try:
            temp.append(storing_values[q-1][pt][1]*bigram_tag_prob[pt][t]*train_emission_prob[t][step])
          except:
            temp.append(storing_values[q-1][pt][1]*0.0001)
        max_temp_index = temp.index(max(temp))
        best_pt = previous_states[max_temp_index]
        storing_values[q][t]=[best_pt,max(temp)]

  #Backtracing to extract the best possible tags for the sentence
  pred_tags = []
  total_steps_num = storing_values.keys()
  last_step_num = max(total_steps_num)
  for bs in range(len(total_steps_num)):
    step_num = last_step_num - bs
    if step_num == last_step_num:
      pred_tags.append('&&')
      pred_tags.append(storing_values[step_num]['&&'][0])
    if step_num<last_step_num and step_num>0:
      pred_tags.append(storing_values[step_num][pred_tags[len(pred_tags)-1]][0])
  predicted_tags.append(list(reversed(pred_tags)))
#Calculating the accuracy based on tagging each word in the test data.
right = 0 
wrong = 0
for i in range(len(test_tags)):
  gt = test_tags[i]
  pred = predicted_tags[i]
  for h in range(len(gt)):
    if gt[h] == pred[h]:
      right = right+1
    else:
      wrong = wrong +1 

print('Accuracy on the test data is: ',right/(right+wrong))
print('Loss on the test data is: ',wrong/(right+wrong))