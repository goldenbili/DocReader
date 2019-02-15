#!/usr/bin/env python3
# Copyright 2018-present, HKUST-KnowComp.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Machine Comprehension predictor"""

import logging

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize

from vector import vectorize, batchify
from model import DocReader
import utils
from spacy_tokenizer import SpacyTokenizer
import time

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Tokenize + annotate
# ------------------------------------------------------------------------------

TOK = None

def init(options):
    global TOK
    TOK = SpacyTokenizer(**options)
    Finalize(TOK, TOK.shutdown, exitpriority=100)


def tokenize(text):
    global TOK
    return TOK.tokenize(text)

def get_annotators_for_model(model):
    annotators = set()
    if model.args.use_pos:
        annotators.add('pos')
    if model.args.use_lemma:
        annotators.add('lemma')
    if model.args.use_ner:
        annotators.add('ner')
    return annotators


# ------------------------------------------------------------------------------
# Predictor class.
# ------------------------------------------------------------------------------

context_tokens = None

class Predictor(object):
    """Load a pretrained DocReader model and predict inputs on the fly."""

    def __init__(self, model, normalize=True,
                 embedding_file=None, char_embedding_file=None, num_workers=None):
        """
        Args:
            model: path to saved model file.
            normalize: squash output score to 0-1 probabilities with a softmax.
            embedding_file: if provided, will expand dictionary to use all
              available pretrained vectors in this file.
            num_workers: number of CPU processes to use to preprocess batches.
        """
        logger.info('Initializing model...')
        self.model = DocReader.load(model, normalize=normalize)

        if embedding_file:
            logger.info('Expanding dictionary...')
            words = utils.index_embedding_words(embedding_file)
            added_words = self.model.expand_dictionary(words)
            self.model.load_embeddings(added_words, embedding_file)
        if char_embedding_file:
            logger.info('Expanding dictionary...')
            chars = utils.index_embedding_chars(char_embedding_file)
            added_chars = self.model.expand_char_dictionary(chars)
            self.model.load_char_embeddings(added_chars, char_embedding_file)

        logger.info('Initializing tokenizer...')
        annotators = get_annotators_for_model(self.model)

        self.workers = None
        self.tokenizer = SpacyTokenizer(annotators=annotators)

    # prerun steps for simplePredict
    def pre(self, document):
        documents = [document]
        if self.workers:
            c_tokens = self.workers.map_async(tokenize, documents)
            c_tokens = list(c_tokens.get())
        else:
            c_tokens = list(map(self.tokenizer.tokenize, documents))
        global context_tokens
        context_tokens = c_tokens

    # only considering one document-question pair (no candidates)
    def simplePredict(self, question, top_n=1):
        questions = [question]

        if self.workers:
            q_tokens = self.workers.map_async(tokenize, questions)
            q_tokens = list(q_tokens.get())
        else:
            q_tokens = list(map(self.tokenizer.tokenize, questions))

        example = {
            'id': 0,
            'question': q_tokens[0].words(),
            'question_char': q_tokens[0].chars(),
            'qlemma': q_tokens[0].lemmas(),
            'qpos': q_tokens[0].pos(),
            'qner': q_tokens[0].entities(),
            'document': context_tokens[0].words(),
            'document_char': context_tokens[0].chars(),
            'clemma': context_tokens[0].lemmas(),
            'cpos': context_tokens[0].pos(),
            'cner': context_tokens[0].entities(),}
            
        # Build the batch and run it through the model
        batch_exs = batchify([vectorize(example, self.model)])
        s, e, score = self.model.predict(batch_exs, None, top_n)

        # Retrieve the predicted spans
        predictions = []
        for j in range(len(s[0])):
            
            # Reconstruct the document by adding * surrounding the predicted answer 
            # Retrieves the sentence(s) containing *answer* rather than answer in the event that answer appears in multiple sentences 
            span = context_tokens[0].slice(s[0][j], e[0][j] + 1).untokenize()
            t0 = time.time()
            doc = context_tokens[0].slice(0,s[0][j]).untokenize() + " /*/" + span + "/*/ " + context_tokens[0].slice(e[0][j] + 1,).untokenize()
            t1 = time.time()
            print(t1-t0)
            predictions.append((span, doc, score[0][j], s[0][j]))

        return [predictions]


    def predict(self, document, question, top_n=1, candidates=None):
        """Predict a single document - question pair."""
        batch = []
        for q in question:
            batch.append([document, q, candidates])
        results = self.predict_batch(batch, top_n)
        return results

    def predict_batch(self, batch, top_n=1):
        """Predict a batch of document - question pairs."""
        documents, questions, candidates = [], [], []
        for b in batch:
            documents.append(b[0])
            questions.append(b[1])
            candidates.append(b[2] if len(b) == 3 else None)
        candidates = candidates if any(candidates) else None

        # Tokenize the inputs, perhaps multi-processed.
        if self.workers:
            q_tokens = self.workers.map_async(tokenize, questions)
            c_tokens = self.workers.map_async(tokenize, documents)
            q_tokens = list(q_tokens.get())
            c_tokens = list(c_tokens.get())
        else:
            q_tokens = list(map(self.tokenizer.tokenize, questions))
            c_tokens = list(map(self.tokenizer.tokenize, documents))

        examples = []
        for i in range(len(questions)):
            examples.append({
                'id': i,
                'question': q_tokens[i].words(),
                'question_char': q_tokens[i].chars(),
                'qlemma': q_tokens[i].lemmas(),
                'qpos': q_tokens[i].pos(),
                'qner': q_tokens[i].entities(),
                'document': c_tokens[i].words(),
                'document_char': c_tokens[i].chars(),
                'clemma': c_tokens[i].lemmas(),
                'cpos': c_tokens[i].pos(),
                'cner': c_tokens[i].entities(),
            })

        # Stick document tokens in candidates for decoding
        if candidates:
            candidates = [{'input': c_tokens[i], 'cands': candidates[i]}
                          for i in range(len(candidates))]

        # Build the batch and run it through the model
        batch_exs = batchify([vectorize(e, self.model) for e in examples])
        s, e, score = self.model.predict(batch_exs, candidates, top_n)

        # Retrieve the predicted spans
        results = []
        for i in range(len(s)):
            predictions = []
            for j in range(len(s[i])):
                span = c_tokens[i].slice(s[i][j], e[i][j] + 1).untokenize()
                doc = context_tokens[0].slice(0, s[i][j]).untokenize() + " /*/" + span + "/*/ " + context_tokens[0].slice(e[i][j] + 1, ).untokenize()
                predictions.append((span, doc, score[i][j], s[i][j]))
                #predictions.append((span, score[i][j]))
            results.append(predictions)
        return results

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()


