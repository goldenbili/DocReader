import sys

sys.path.append('.')


import threading
from time import localtime
import time

import torch
import torch._utils
STOPWORDS_ATEN = frozenset(['device','use','KE6900','ke6940','ke8950','ke8952','ke69xx','ke89xx','work','purpose','cannot','problem'])

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor


    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
import code
import argparse
import logging
import prettytable
import json
import os
import imp
import re

from predictor import Predictor
from multiprocessing import cpu_count

import nltk
import nltk.data
nltk.download('punkt')


logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

# speeds up cuda
torch.backends.cudnn.benchmark = True

dir = os.getcwd()

'''
PREDICTOR = Predictor(
        dir + "/data/models/20181003-cd2d0c87.mdl",
        normalize=True,
        embedding_file=None,
        char_embedding_file=None,
        num_workers=2,
)
'''
# ------------------------------------------------------------------------------
# Drop in to interactive mode
# ------------------------------------------------------------------------------

import numpy as np
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

'''
# elvis file for testing
elvis = wikipedia.page("Elvis Presley").content

# removing headers from Elvis file
import re
elvis = re.sub(r'\=.*?\=', '', elvis)
elvis = elvis.split('\n\n')
'''

# willy add : read and decode QA_database.txt
rdIntentList = [
    "intentQ1", "intentQ2", "intentQ3", "intentQ4", "intentQ5", "intentQ6", "intentQ7", "intentQ8", "intentQ9",
    "intentQ10", "intentQ11", "intentQ12", "intentQ13", "intentQ14", "intentQ15", "intentQ16", "intentQ17", "intentQ18",
    "intentQ19",
    "intentQ20", "intentQ21", "intentQ22", "intentQ23", "intentQ24", "intentQ25", "intentQ26", "intentQ27", "intentQ28",
    "intentQ29",
    "intentQ30", "intentQ31", "intentQ32", "intentQ33", "intentQ34", "intentQ35", "intentQ36", "intentQ37", "intentQ38",
    "intentQ39", "intentQ40"
]


def read_intent_name(intent_name, file_ptr):
    line = file_ptr.readline()
    intent_list = []

    while line:
        line = line.strip()
        cmp_str = "[Intent:]"
        # cmp_str = cmp_str.encode("utf-8")
        if line == cmp_str:
            line = file_ptr.readline()
            line = line.strip()
            if intent_name == line:
                # not do yet ... (WillyChang20190122)
                break
                # print(line)
        line = file_ptr.readline()
    return intent_list


def read_all_intent(file_ptr):
    line = file_ptr.readline()
    intent_list = []

    idx = 0
    while line:
        line = line.strip()
        cmp_str = "[Intent:]"
        # cmp_str = cmp_str.encode("utf-8")
        if line == cmp_str:
            line = file_ptr.readline()
            line = line.strip()
            intent_list.append([])
            if 'intentQ' in line:
                while line != '[End]':
                    line = file_ptr.readline()
                    line = line.strip()
                    if (line != '[End]'):
                        intent_list[idx].append(line.strip())
                idx = idx + 1
            else:
                print('this is an QAData format error ...(willychang)\n')
                # print(line)
        line = file_ptr.readline()
    return intent_list


# Kiite documents from "Everyone" drive in Google Docs converted to plain text
# Place holder way of retreiving this data until pipeline is built
'''
import pickle
dir = "/Users/mandygu/Desktop/Testing Kiite/drives"
with open(dir+"/text.pkl", "rb") as handle:   
   kiite = pickle.load(handle)

for i in range(0, len(kiite)):
    kiite[i] = kiite[i].decode('UTF-8')
'''
import nltk
import nltk.data


# Tokenizes text after dropping stop words and converting to lowercase
# Short tokens less than length of 2 are dropped

def tokenize(text):
    return [token.lower() for token in simple_preprocess(text) if token not in STOPWORDS]


# Returns the full sentence which contains the answer from the document
# Uses a pretrained tokenizer to break text into sentences

def returnSentence(document, answer):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(document)
    length = len(sentences)

    # if answer is within subset of a sentence, return the entire sentence
    for i in range(0, length):
        if answer in sentences[i]:
            return sentences[i]

    # if answer spreads over two sentences (rare), return both sentences
    for i in range(0, length - 1):
        if answer in sentences[i] + " " + sentences[i + 1]:
            return sentences[i] + " " + sentences[i + 1]
        elif answer in sentences[i] + sentences[i + 1]:
            return sentences[i] + sentences[i + 1]

    # if answer spreads over more than two sentences (very rare)
    answer = ""

    for sentence in sentences:
        if "/*/" in sentence or sentence in answer:
            answer = answer + " " + sentence

    return answer
    '''
    for i in range(0, length): 
        if answer in sentences[i]: 
            return sentences[i].replace("/*/","")

    # if answer spreads over two sentences (rare), return both sentences
    for i in range(0, length -1): 
        if answer in sentences[i] + " " + sentences[i+1]:
            return sentences[i].replace("/*/","") + " " + sentences[i+1].replace("/*/","")
        elif answer in sentences[i] + sentences[i+1]:
            return sentences[i].replace("/*/","") + sentences[i + 1].replace("/*/","")

    # if answer spreads over more than two sentences (very rare)
    answer = ""

    for sentence in sentences: 
        if "/*/" in sentence or sentence in answer: 
            answer = answer + " " + sentence

    return answer.replace("/*/","")
    '''


# Preprocessing to determine candidate regions

def preprocess(text, question, n):
    t0 = time.time()

    # process text by tokenizing and removing non-alphabetical characters
    length = len(text)
    processedParagraphs = [None] * length
    for i in range(0, length):
        processedParagraphs[i] = tokenize(text[i])
        paragraphLength = len(processedParagraphs[i])
        for x in range(0, paragraphLength):
            processedParagraphs[i][x] = "".join([char for char in processedParagraphs[i][x] if char.isalpha()])
    processedParagraphs = [" ".join(x) for x in processedParagraphs]

    # No dimensionality reduction in this step
    # Used to obtain singular values of all coordinates to determine optimal number of components to use
    vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True)
    svdModel = TruncatedSVD(n_components=length)
    vectorizedParagraphs = vectorizer.fit_transform(processedParagraphs)
    svdModel.fit_transform(vectorizedParagraphs)

    singularValues = svdModel.singular_values_

    # Use augmented Kaiser's Stopping Rule: choose all components with singular values > 0.8
    num = sum(1 for sv in singularValues if sv > 0.8)

    if num < 10:
        num = length

        # Recreate preprocessor which reduces to num dimensions
    svdModel = TruncatedSVD(n_components=num)
    svdModel.fit_transform(vectorizedParagraphs)

    # transform the question using same vectorizer and SVD
    t0 = time.time()
    tokenizedQuestion = " ".join(tokenize(question))
    vectorizedQuestion = vectorizer.transform([tokenizedQuestion])
    processedQuestion = svdModel.transform(vectorizedQuestion)

    confidence = []

    # Use cosine similarity to determine closest regions to question
    for i in range(0, length):
        vectorizedParagraph = vectorizer.transform([processedParagraphs[i]])
        paragraph = svdModel.transform(vectorizedParagraph)
        sim = cosine_similarity(processedQuestion, paragraph)
        confidence.append([i, sim])

    confidence.sort(key=lambda x: x[1], reverse=True)

    newDocument = ""

    # combines top n regions
    for i in confidence[:n]:
        newDocument += text[i[0]] + "\n\n."

    t1 = time.time()

    print("Time used to locate top N candidates: ", t1 - t0)

    return [confidence, newDocument]


# Combines preprocessor with Mnemonic Reader
# Set first to False if not first time processing document; this prevents the same preprocessing steps from being ran again
# n1 is number of candidate files to choose
# n2 is number of candidate paragraphs to choose from candidate files
'''
def processQuestion(question, text = elvis, n1 = 8, n2 = 15, first = True):

    # find the top n1 candidate files 
    preprocessFiles = preprocess(text, question, n1)
    candidateFiles = preprocessFiles[1]

    # confidence scores for these files 
    confidence = preprocessFiles[0]

    paragraphs = candidateFiles.split("\n\n")

    # find the top n2 candidate paragraphs
    candidateParagraphs = preprocess(paragraphs, question, n2)[1]

    # Runs Mnemonic Reader over the top paragraphs
    prediction = process(candidateParagraphs, question, first)

    return [confidence, prediction]
'''


# precompiles the preprocessing steps for Mnemonic Reader

def precompile(document):
    PREDICTOR.pre(document)


# original prediction function for Mnemonic Reader
# uses the precompile function to reduce latency

# def process(document, question,first, candidates=None, top_n=1):

def process(question, first, candidates=None, top_n=1):
    t0 = time.time()

    file = open("Output1.txt", "r")
    document = file.read()
    file.close()
    precompile(document)

    questions = []
    questions.append(question)
    '''
    if'KE6900' in question:
        questions.append(question.replace('KE6900', 'KE69 Series'))
        questions.append(question.replace('KE6900', 'KE Series'))
    elif 'KE6940' in question:
        questions.append(question.replace('KE6940', 'KE69 Series'))
        questions.append(question.replace('KE6940', 'KE Series'))
    elif 'KE8950' in question:
        questions.append(question.replace('KE8950', 'KE89 Series'))
        questions.append(question.replace('KE8950', 'KE Series'))
    elif 'KE8952' in question:
        questions.append(question.replace('KE8952', 'KE89 Series'))
        questions.append(question.replace('KE8952', 'KE Series'))
    '''
    # predictions = PREDICTOR.simplePredict(question, top_n)
    predictions = PREDICTOR.predict(document, questions, top_n)
    # predictions = predictions[0]
    table = prettytable.PrettyTable(["Rank", "Span", "Score"])

    anslist = []
    spanlist = []
    for preds in predictions:
        for i, p in enumerate(preds, 1):
            prediction = '/*/' + p[0] + '/*/'
            doc = p[1]
            answer = returnSentence(doc, prediction)
            table.add_row([i, answer, p[2]])

            spanlist.append(p[0])
            answer = answer.replace("/*/", "")
            anslist.append(answer)

            '''
            answer = answer.replace("/*/","")
            if len(anslist)==0:
                anslist.append(answer)
            else:
                isSet = False
                answereduce = replaceMultiple(answer, [',', '.', ' '], "")
                for i, ans in enumerate(anslist):
                    ansreduce = replaceMultiple(ans, [',', '.', ' '], "")
                    if ansreduce == answereduce or answereduce in ansreduce:
                        isSet = True
                        break
                    elif ansreduce in answereduce:
                        isSet = True
                        anslist[i] = answer
                        break
                if isSet==False:
                    anslist.append(answer)
            '''

    t1 = time.time()
    print(table)
    print("Time for MR to make prediction: ", t1 - t0)

    return [anslist, spanlist]

    '''
    t0 = time.time()
    file = open("Output2.txt", "r")
    document = file.read()
    file.close()
    if first:
        precompile(document)
    predictions = PREDICTOR.simplePredict(question, top_n)
    doc = predictions[1]
    predictions = predictions[0]
    table = prettytable.PrettyTable(["Rank", "Span", "Score"])
    for i, p in enumerate(predictions, 1):
        prediction = '/*/'+p[0]+'/*/'
        answer = returnSentence(doc, prediction)
        table.add_row([i, answer, p[1]])
    t1 = time.time()
    print(table)
    print("Time for MR to make prediction: ", t1 - t0)

    t0 = time.time()
    file = open("Output3.txt", "r")
    document = file.read()
    file.close()
    if first:
        precompile(document)
    predictions = PREDICTOR.simplePredict(question, top_n)
    doc = predictions[1]
    predictions = predictions[0]
    table = prettytable.PrettyTable(["Rank", "Span", "Score"])
    for i, p in enumerate(predictions, 1):
        prediction = '/*/'+p[0]+'/*/'
        answer = returnSentence(doc, prediction)
        table.add_row([i, answer, p[1]])
    t1 = time.time()
    print(table)
    print("Time for MR to make prediction: ", t1 - t0)

    t0 = time.time()
    file = open("Output4.txt", "r")
    document = file.read()
    file.close()
    if first:
        precompile(document)
    predictions = PREDICTOR.simplePredict(question, top_n)
    doc = predictions[1]
    predictions = predictions[0]
    table = prettytable.PrettyTable(["Rank", "Span", "Score"])
    for i, p in enumerate(predictions, 1):
        prediction = '/*/'+p[0]+'/*/'
        answer = returnSentence(doc, prediction)
        table.add_row([i, answer, p[1]])

    t1 = time.time()
    print(table)
    print("Time for MR to make prediction: ", t1 - t0)
    '''

    # return predictions


# ------------------------------------------------------------------------------
# Testing Mnemonic Reader
# ------------------------------------------------------------------------------

# Tests preprocessor accuracy
# takes a list of questions with the correct paragraph the ground truth answer is found in
# Outputs number of times the correct answer is found in the top num paragraphs
'''
def testPreprocessor(questions, n1, n2, doc = elvis):
    n = len(questions)
    naiveScore = 0
    ignoreScore = 0 
    ignoreError = 0
    cosine = []
    for i in range(0,n):
        correctParagraph = questions[i][2]
        ignore = questions[i][1]
        result = processQuestion(questions[i][0], doc, n1, n2)[0]
        topParagraphs = [i[0] for i in result]
        if correctParagraph in topParagraphs[:n1]:
            naiveScore += 1
        confidence = result[0][1][0][0]
        if confidence <= 0.05 and ignore == True: 
            ignoreScore += 1
        if confidence <= 0.05 and ignore == False: 
            ignoreError += 1
        cosine.append(confidence)
    print("no. correct: ", naiveScore, " || no. correctly ignored: , ", ignoreScore, " || no. incorrectly ignored: ",ignoreError, " || no. total: ", n)
    print(cosine)

# Tests the combination of the preprocessor and Mnemonic Reader for accuracy by counting the number of Exact Matches and Partial Matches 
# Exact match is when Mnemonic Reader's answer is exactly the same as the human answer 
# Partial Match is when the human answer is a subset of Mnemonic Reader's answer

def testMR(questions, answers,n1, n2, doc = elvis):
    n = len(questions)
    results = {}
    exactMatch = 0
    partialMatch = 0
    for i in range(0,n):
        question = questions[i][0]
        qType = questions[i][1]
        inText = questions[i][2]
        answer = answers[i]
        results[question] = [] 
        mOutput = processQuestion(question, doc, n1, n2)[1][0]
        mAnswer = mOutput[0]
        confidence = mOutput[1]
        if mAnswer == answer: 
            match = "exact match"
            exactMatch += 1
        elif answer in mAnswer: 
            match = "partial match"
            partialMatch += 1
        else: 
            match = "no match"
        result = {"answer":answer, "MR answer": mAnswer, "confidence": int(round(confidence*100)), 
                    "match": match, "Question type": qType, "In text": inText, "Num Paragraphs": n2}
        results[question].append(result)
    print("Exact Match: ", exactMatch, " || Partial Match: ", partialMatch, " || no. total: ", n)
    with open('testResults.json', 'w') as f:
        json.dump(results, f)
'''

Apotoken = ['!', '\"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?',
            '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', ',']


def replaceMultiple(mainString, toBeReplaces, newString):
    # Iterate over the strings to be replaced
    for elem in toBeReplaces:
        # Check if string is in the main string
        if elem in mainString:
            # Replace the string
            mainString = mainString.replace(elem, newString)

    return mainString


banner = """
* WRMCQA interactive Document Reader Module *

* Repo: Mnemonic Reader (https://github.com/HKUST-KnowComp/MnemonicReader)

* Implement based on Facebook's DrQA

>>> process(document, question, candidates=None, top_n=1)
>>> usage()

* To test with preprocessor 
- change first to False if not first time querying from the text
- default text is wikipedia elvis corpus

>>> processQuestion(question, text = elvis, n1, n2, first = True)

"""


def usage():
    print(banner)

# ------------------------------------------------------------------------------
# Commandline arguments & init
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="data/models/20181003-cd2d0c87.mdl",
                        help='Path to model to use')
    parser.add_argument('--embedding-file', type=str, default=None,
                        help=('Expand dictionary to use all pretrained '
                              'embeddings in this file.'))
    parser.add_argument('--char-embedding-file', type=str, default=None,
                        help=('Expand dictionary to use all pretrained '
                              'char embeddings in this file.'))
    parser.add_argument('--num-workers', type=int, default=int(cpu_count() / 2),
                        help='Number of CPU processes (for tokenizing, etc)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Use CPU only')
    parser.add_argument('--no-readintent', action='store_true',
                        help='Use read intent token')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='Specify GPU device id to use')
    parser.add_argument('--no-normalize', action='store_true',
                        help='Do not softmax normalize output scores.')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        logger.info('CUDA enabled (GPU %d)' % args.gpu)
    else:
        logger.info('Running on CPU only.')

    PREDICTOR = Predictor(
        args.model,
        normalize=not args.no_normalize,
        embedding_file=args.embedding_file,
        char_embedding_file=args.char_embedding_file,
        num_workers=args.num_workers,
    )
    if args.cuda:
        PREDICTOR.cuda()

    fp = open('QA_database.txt', "r")
    intent_list = read_all_intent(fp)
    for i in range(len(intent_list)):
        Sintent_idx = '========================(intentQ' + str(i + 1) + ')========================'
        print('%s' %(Sintent_idx))
        for j, ques in enumerate(intent_list[i]):
            print('question %d - %d : %s' % (i+1, j, ques))


            anslist = process(ques, True, candidates=None, top_n=3)

            # do 1a2b
            # ------------------------------------------------------------------------------
            myMessage = replaceMultiple(ques, Apotoken, '')

            quesToken = myMessage.lower().split(' ')
            pointList = []
            for m, ans in enumerate(anslist[0], 1):
                point = 0
                ans_low = ans.lower()
                ans_low = replaceMultiple(ans_low, Apotoken, '')
                for quesOneWord in enumerate(quesToken, 1):
                    if quesOneWord[1] != '':
                        tokenIndex = [m.start() for m in re.finditer(quesOneWord[1], ans_low)]
                        if (len(tokenIndex) != 0 and quesOneWord[1] not in STOPWORDS and quesOneWord[1] not in STOPWORDS_ATEN):
                            point = point + 1
                pointList.append(point)

            indexAns = 0
            for k, p in enumerate(pointList, 1):
                if (p > pointList[indexAns]):
                    indexAns = k - 1
            # ------------------------------------------------------------------------------

            print('Dr_Answer: %s' %(anslist[1][indexAns]))
            print('Dr_QA: %s' %(anslist[0][indexAns]))
            
    print('DataSet is finish')
    '''
    file = open("Output1.txt", "r")
    document = file.read()
    file.close()
    precompile(document)
    code.interact(banner=banner, local=locals())
    '''
    '''
    tserver = None
    while tserver == None:
        tserver = TcpServer(args.no_readintent)
    tserver.listen_client()
    sys.exit()
    '''
