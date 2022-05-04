import nltk
from nltk.tag import brill, brill_trainer
from nltk.tag import DefaultTagger
tag = DefaultTagger('DT')
# initializing training and testing set    
x = open("article_1.txt","r")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
tokens = nltk.word_tokenize(x.read())
final_tags = []
for i in tokens:
    final_tags.extend(nltk.pos_tag(nltk.word_tokenize(i)))
print(final_tags[:2])
tags = [final_tags]
split_num = int(len(final_tags)*0.9)
print(split_num)
train = final_tags[:split_num]
test = final_tags[split_num:]
print(test)
def brill_tagger(tag, train, **kwargs):

    templates = [

            brill.Template(brill.Pos([2])),

            brill.Template(brill.Pos([-2, -1])),

            brill.Template(brill.Pos([1, 2])),

            brill.Template(brill.Pos([-3, -2, -1])),

            brill.Template(brill.Pos([1, 2, 3])),

            brill.Template(brill.Pos([-1]), brill.Pos([1])),

            brill.Template(brill.Word([1])),

            brill.Template(brill.Word([-2])),

            brill.Template(brill.Word([-2, -1])),

            brill.Template(brill.Word([1, 2])),

            brill.Template(brill.Word([-3, -2, -1])),

            brill.Template(brill.Word([1, 2, 3])),

            brill.Template(brill.Word([-1]), brill.Word([1])),]
    f = brill_trainer.BrillTaggerTrainer(tag, templates, deterministic = True)

    return f.train(train, **kwargs)

brillz = brill_tagger(tag, train)
b = brillz.evaluate(test)

print ("Accuracy of brill tag : ", b)