# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
import codecs
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import GRU
import numpy as np
import re
from sklearn.metrics import confusion_matrix
from datetime import datetime
from keras.layers.wrappers import Bidirectional
import os.path

# ########################################Constants###########################################################
'''
    Experiment Configuration:
'''
EXPERIMENT_CONFIGURATION = '1_Bidirectional_without_Dropout_optimizer_rmsprop'
maxlen = 65
maxlen_after = 10

SAKEN_CHAR = 'ْ'

FATHE_TASHDID = 'َّ'
KASRE_TASHDID = 'ِّ'
ZAMME_TASHDID = 'ُّ'
TANVINE_FATHE_TASHDID = 'ًّ'
TANVINE_KASREH_TASHDID = 'ٍّ'
TANVINE_ZAMME_TASHDID = 'ٌّ'

TASHDID_FATHE = 'َّ'
TASHDID_KASRE = 'ِّ'
TASHDID_ZAMME = 'ُّ'
TASHDID_TANVINE_FATHE = 'ًّ'
TASHDID_TANVINE_KASRE = 'ٍّ'
TASHDID_TANVINE_ZAMME = 'ٌّ'
TASHDID = 'ّ'

diacritics_mappings = {TASHDID_FATHE: '@', TASHDID_KASRE: '#', TASHDID_ZAMME: '$',
                       TASHDID_TANVINE_FATHE: '&', TASHDID_TANVINE_KASRE: 'α', TASHDID_TANVINE_ZAMME: 'β'}

diacritics = set(['ّ', 'ْ', 'ٌ', 'ٍ', 'ً', 'ُ', 'ِ', 'َ', '@', '#', '$', '&', 'α', 'β'])

# ################################################Preparing Training and Testing Data###############################
# Original Text
original_text = ''
if os.path.exists('original_text.txt'):
    with codecs.open('original_text.txt', encoding='utf-8-sig') as original_text_file:
        original_text = original_text_file.read()
else:
    with codecs.open('Arabic_Training_Set_Revised1.txt', encoding='utf-8-sig') as myfile:
        text = myfile.read()

    original_text = text.strip()
    original_text = re.sub("[0-9a-zA-Z@#$&αβ()\[\]{}~*;,?\"\-_]", "", original_text)

    original_text = original_text.replace('!', '')
    original_text = original_text.replace(':', '')

    original_text = original_text.replace(SAKEN_CHAR + SAKEN_CHAR, SAKEN_CHAR)  # Remove excessive SAAKENS
    original_text = original_text.replace('ََ', 'َ')
    original_text = original_text.replace('ُُ', 'ُ')
    original_text = original_text.replace('ِِ', 'ِ')
    original_text = original_text.replace(FATHE_TASHDID + 'َ', FATHE_TASHDID)
    original_text = original_text.replace(KASRE_TASHDID + 'ِ', KASRE_TASHDID)
    original_text = original_text.replace(ZAMME_TASHDID + 'ُ', ZAMME_TASHDID)

    with codecs.open('original_text.txt', encoding='utf-8', mode='w+') as f:
        f.write(original_text)


print('corpus length:', len(original_text))
# print('text:', repr(text))

'''
    create two sets of data: training: 75% of data and testing: 25% of data
'''

original_text = original_text#[0:1000]
original_text_length = len(original_text)

# Refined Text
refined_text = ''
training_data = ''
testing_data = ''
refined_testing_data = ''
if os.path.exists('refined_text.txt'):
    with codecs.open('refined_text.txt', encoding='utf-8-sig') as refined_text_file:
        refined_text = refined_text_file.read()
else:
    training_data = original_text[: int(0.75 * original_text_length)].strip()


    # make sure training and testing data do not start with diacritics
    for d in diacritics:
        testing_data = testing_data.lstrip(d)
        training_data = training_data.lstrip(d)

    training_data = training_data.replace(FATHE_TASHDID, TASHDID_FATHE)
    training_data = training_data.replace(KASRE_TASHDID, TASHDID_KASRE)
    training_data = training_data.replace(ZAMME_TASHDID, TASHDID_ZAMME)
    training_data = training_data.replace(TANVINE_FATHE_TASHDID, TASHDID_TANVINE_FATHE)
    training_data = training_data.replace(TANVINE_KASREH_TASHDID, TASHDID_TANVINE_KASRE)
    training_data = training_data.replace(TANVINE_ZAMME_TASHDID, TASHDID_TANVINE_ZAMME)

    for k, v in diacritics_mappings.items():
        training_data = training_data.replace(k, v)

    # ############################Cleaning Training Data###############################
    splitted_training_data = re.split('[!؟:،؛,.]', training_data)
    cleaned_training_data = ''

    for s in splitted_training_data:
        if len(s) > maxlen:
            cleaned_training_data += s + '.'

    refined_text = ''
    for i in range(0, len(cleaned_training_data) - 1):
        if cleaned_training_data[i] in diacritics:
            refined_text += cleaned_training_data[i]
        elif cleaned_training_data[i + 1] not in diacritics:
            refined_text += cleaned_training_data[i] + SAKEN_CHAR
        else:
            refined_text += cleaned_training_data[i]

    i += 1
    if cleaned_training_data[i] in diacritics:
        refined_text += cleaned_training_data[i]
    else:
        refined_text += cleaned_training_data[i] + SAKEN_CHAR

    # print('refined_text', refined_text)
    with codecs.open('refined_text.txt', encoding='utf-8', mode='w+') as f:
        f.write(refined_text)



########################################################################
'''
    do it again for testing data for calculating accuracy at the end
'''

# Testing Data & Refined Testing Data
testing_data = original_text[int(0.75 * original_text_length):].strip()
if os.path.exists('testing_data.txt') & os.path.exists('refined_testing_data'):
    with codecs.open('testing_data.txt', encoding='utf-8-sig') as testing_data_file:
        testing_data = testing_data_file.read()

    with codecs.open('refined_testing_data.txt', encoding='utf-8-sig') as refined_testing_data_file:
        refined_testing_data = refined_testing_data_file.read()
else:
    testing_data = testing_data.replace(FATHE_TASHDID, TASHDID_FATHE)
    testing_data = testing_data.replace(KASRE_TASHDID, TASHDID_KASRE)
    testing_data = testing_data.replace(ZAMME_TASHDID, TASHDID_ZAMME)
    testing_data = testing_data.replace(TANVINE_FATHE_TASHDID, TASHDID_TANVINE_FATHE)
    testing_data = testing_data.replace(TANVINE_KASREH_TASHDID, TASHDID_TANVINE_KASRE)
    testing_data = testing_data.replace(TANVINE_ZAMME_TASHDID, TASHDID_TANVINE_ZAMME)

    for k, v in diacritics_mappings.items():
        testing_data = testing_data.replace(k, v)

    ########################################################################################
    refined_testing_data = ''

    for i in range(0, len(testing_data) - 1):
        if testing_data[i] in diacritics:
            refined_testing_data += testing_data[i]
        elif testing_data[i + 1] not in diacritics:
            refined_testing_data += testing_data[i] + SAKEN_CHAR
        else:
            refined_testing_data += testing_data[i]
    i += 1
    if testing_data[i] in diacritics:
        refined_testing_data += testing_data[i]
    else:
        refined_testing_data += testing_data[i] + SAKEN_CHAR

    refined_testing_data = refined_testing_data.lstrip(''.join(diacritics))


    # testing_data = refined_testing_data  # think about if it is correct to add this line
    # preparing testing data
    for c in diacritics:
        testing_data = testing_data.replace(c, '')

    testing_data = ((maxlen - 1) // 2) * (' ' + SAKEN_CHAR) + testing_data

    # print("testing_data", testing_data)

    # print('refined_testing_data', refined_testing_data)

    with codecs.open('testing_data.txt', encoding='utf-8', mode='w+') as f:
        f.write(testing_data)

    with codecs.open('refined_testing_data.txt', encoding='utf-8', mode='w+') as f:
        f.write(refined_testing_data)

diacritics_dict = dict((c, 0) for i, c in enumerate(np.array(list(diacritics))))
print(diacritics_dict)
for c in refined_testing_data:
    if c in diacritics_dict:
        diacritics_dict[c] += 1
print('Number of occurrence of each diacritic:')
print(diacritics_dict)

#####################################################################################################

alphabet = set('شسزرذدخحجثتةبائؤأإآءیۀگکژچپيوهنملقفغعظطضصٔٴ ك ى') - set(' ')
all_chars = set(refined_text) | set(original_text) | diacritics | alphabet
characters_without_need_to_diacritics = (all_chars - diacritics) - alphabet

print('all_chars:', all_chars)
print('diacritics:', diacritics)
print('alphabet:', alphabet)
print('characters_without_need_to_diacritics:', characters_without_need_to_diacritics)
print('total chars:', len(all_chars))
print('diacritics:', len(diacritics))
print('characters_without_need_to_diacritics:', len(characters_without_need_to_diacritics))
print('alphabet:', len(alphabet))

char_indices = dict((c, i) for i, c in enumerate(all_chars))
indices_char = dict((i, c) for i, c in enumerate(all_chars))
diacritics_indices = dict((c, i) for i, c in enumerate(diacritics))
indices_diacritics = dict((i, c) for i, c in enumerate(diacritics))
######################################################################################################

# cut the text in semi-redundant sequences of maxlen characters

step = 2
sentences = []
next_chars = []

refined_text_tokens = re.split('[.]', refined_text)
for token in refined_text_tokens:
    token = token.lstrip(''.join(diacritics))
    token_length = len(token)
    token += maxlen_after * (' ' + SAKEN_CHAR)
    for i in range(0, token_length - maxlen, step):
        rx = '[' + re.escape(''.join(diacritics)) + ']'
        after_chars = re.sub(rx, '', token[i + maxlen + 1: i + maxlen + 1 + (2 * maxlen_after)])
        sentences.append(token[i: i + maxlen] + after_chars)
        next_chars.append(token[i + maxlen])


print('nb sequences:', len(sentences))
print('Vectorization...')
X = np.zeros((len(sentences), maxlen + maxlen_after, len(all_chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(diacritics)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, diacritics_indices[next_chars[i]]] = 1

with codecs.open('inputs.txt', encoding='utf-8', mode='w+') as f:
    for i, sentence in enumerate(sentences):
        f.write((repr(i) + '\n'))
        f.write(('\nsentence:' + sentence + '\n'))
        f.write(('\nnext_char:' + next_chars[i] + '\n'))
        f.write('\n')

# build the model: 2 stacked LSTM
print('Build model...')

nn = 32

nn=64
model = Sequential()
model.add(GRU(nn*4, return_sequences=True, input_shape=(maxlen + maxlen_after, len(all_chars))))
model.add(Dropout(0.10))
model.add(GRU(nn*3, return_sequences=True))
model.add(Dropout(0.10))
#model.add(GRU(nn*2, return_sequences=True))
#model.add(Dropout(0.10))
model.add(GRU(nn))
model.add(Dropout(0.10))
model.add(Dense((len(diacritics))))
model.add(Activation('softmax'))

optimizer = 'rmsprop'
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    # preds = np.asarray(preds).astype('float64')
    return np.argmax(preds)
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
testing_data_length = len(testing_data)
testing_data += maxlen_after * ' '
refined_testing_data += maxlen_after * (' ' + SAKEN_CHAR)

statistics_result = ''
accuracy_in_first_80_chars = 0.0
number_of_epoch = 10
for iteration in range(1, 31):
    with codecs.open(EXPERIMENT_CONFIGURATION + '_Output.txt', encoding="utf-8", mode='a') as f:
        f.write('\n')
        f.write(('-' * 50) + '\n')
        f.write('\nIteration ' + str(iteration) + '\n')
        f.write('\nStart time: ' + str(datetime.now()) + '\n')

    model.fit(X, y, batch_size=nn*10, nb_epoch=number_of_epoch)

    for diversity in [1.0]:
        with codecs.open(EXPERIMENT_CONFIGURATION + '_Output.txt', encoding="utf-8", mode='a') as f:
            f.write('\n')
            f.write('----- diversity: ' + str(diversity) + '\n')

        next_diacritic = ''
        generated = testing_data[0: maxlen - 1]
        sentence = ''

        correct_prediction = 0.0
        wrong_prediction = 0.0
        refined_testing_data_index = 0
        y_true = []
        y_pred = []
        for i in range(maxlen - 1, testing_data_length):
            sentence = generated[len(generated) - maxlen + 1:] + testing_data[i] + testing_data[i + 1: i + 1 + maxlen_after]

            x = np.zeros((1, maxlen + maxlen_after, len(all_chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_diacritic = indices_diacritics[next_index]

            refined_testing_data_index = 2 * (i - (maxlen - 1)) + 1
            if next_diacritic == refined_testing_data[refined_testing_data_index]:   # finding out is the prediction correct
                correct_prediction += 1.0                                              # in the training data
            else:
                wrong_prediction += 1.0

            y_true.append(refined_testing_data[refined_testing_data_index])
            y_pred.append(next_diacritic)

            generated += testing_data[i] + next_diacritic

            if i == 80 + (maxlen - 1):
                accuracy_in_first_80_chars = round((correct_prediction/(correct_prediction + wrong_prediction)) * 100, 4)

        generated = generated.replace(SAKEN_CHAR, '')
        for k, v in diacritics_mappings.items():
            generated = generated.replace(v, k)

        with codecs.open(EXPERIMENT_CONFIGURATION + '_Output.txt', encoding='utf-8-sig', mode='a') as f:
            f.write('test data with diacritics: \n' + generated)

        statistics_result = 'Iteration ' + str(iteration)
        statistics_result += '\nNumber of Epoch: ' + str(number_of_epoch)
        statistics_result += '\nOptimizer: ' + optimizer
        statistics_result += '\nNumber of Layer: ' + str(model.layers.count(Bidirectional))  #does not work
        statistics_result += '\nnn: ' + str(nn)
        statistics_result += '\nDiversity: ' + str(diversity)
        statistics_result += '\nCorrect Predictions: ' + str(correct_prediction)
        statistics_result += '\nWrong Predictions: ' + str(wrong_prediction)
        statistics_result += '\nAccuracy in first 80 characters: ' + \
                             str(accuracy_in_first_80_chars) + ' %'
        statistics_result += '\nAccuracy in Entire Testing data: ' + \
                             str(round((correct_prediction/(correct_prediction + wrong_prediction)) * 100, 4)) + ' %'

        labels = np.array(list(diacritics))
        statistics_result += '\nnp.array(list(diacritics)): \n' + str(labels)

        confusion_matrix_output = confusion_matrix(y_true, y_pred, labels=labels)
        statistics_result += '\nConfusion_matrix: \n' + str(confusion_matrix_output)

        confusion_matrix_percentile = ''
        for i in range(confusion_matrix_output.shape[0]):
            s = sum(confusion_matrix_output[i])
            confusion_matrix_percentile += str(([round(float(j)/s, 3) for j in confusion_matrix_output[i]])) + '\n'

        statistics_result += '\nconfusion_matrix_percentile: \n' + confusion_matrix_percentile
        statistics_result += str('\n' + ('*' * 50) + '\n\n')

        with codecs.open(EXPERIMENT_CONFIGURATION + '_Statistics.txt', encoding='utf-8-sig', mode='a') as f:
            f.write(statistics_result)

        with codecs.open(EXPERIMENT_CONFIGURATION + '_Output.txt', encoding='utf-8-sig', mode='a') as f:
            f.write('\nEnd time: ' + str(datetime.now()) + '\n' + '\n')


        # model.save(EXPERIMENT_CONFIGURATION + '_Model.h5', overwrite=True)

model.save(EXPERIMENT_CONFIGURATION + '_Model.h5', overwrite=True)
