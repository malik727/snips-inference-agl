from __future__ import division, unicode_literals

import itertools
import re
from builtins import next, range, str
from copy import deepcopy
from uuid import uuid4

from future.utils import iteritems, itervalues

from nlu_inference_agl.constants import (DATA, ENTITY, INTENTS, TEXT,
                                 UNKNOWNWORD, UTTERANCES)
from nlu_inference_agl.data_augmentation import augment_utterances
from nlu_inference_agl.dataset import get_text_from_chunks
from nlu_inference_agl.entity_parser.builtin_entity_parser import is_builtin_entity
from nlu_inference_agl.preprocessing import tokenize_light
from nlu_inference_agl.resources import get_noise

NOISE_NAME = str(uuid4())
WORD_REGEX = re.compile(r"\w+(\s+\w+)*")
UNKNOWNWORD_REGEX = re.compile(r"%s(\s+%s)*" % (UNKNOWNWORD, UNKNOWNWORD))


def get_noise_it(noise, mean_length, std_length, random_state):
    it = itertools.cycle(noise)
    while True:
        noise_length = int(random_state.normal(mean_length, std_length))
        # pylint: disable=stop-iteration-return
        yield " ".join(next(it) for _ in range(noise_length))
        # pylint: enable=stop-iteration-return


def generate_smart_noise(noise, augmented_utterances, replacement_string,
                         language):
    text_utterances = [get_text_from_chunks(u[DATA])
                       for u in augmented_utterances]
    vocab = [w for u in text_utterances for w in tokenize_light(u, language)]
    vocab = set(vocab)
    return [w if w in vocab else replacement_string for w in noise]


def generate_noise_utterances(augmented_utterances, noise, num_intents,
                              data_augmentation_config, language,
                              random_state):
    import numpy as np

    if not augmented_utterances or not num_intents:
        return []
    avg_num_utterances = len(augmented_utterances) / float(num_intents)
    if data_augmentation_config.unknown_words_replacement_string is not None:
        noise = generate_smart_noise(
            noise, augmented_utterances,
            data_augmentation_config.unknown_words_replacement_string,
            language)

    noise_size = min(
        int(data_augmentation_config.noise_factor * avg_num_utterances),
        len(noise))
    utterances_lengths = [
        len(tokenize_light(get_text_from_chunks(u[DATA]), language))
        for u in augmented_utterances]
    mean_utterances_length = np.mean(utterances_lengths)
    std_utterances_length = np.std(utterances_lengths)
    noise_it = get_noise_it(noise, mean_utterances_length,
                            std_utterances_length, random_state)
    # Remove duplicate 'unknownword unknownword'
    return [
        text_to_utterance(UNKNOWNWORD_REGEX.sub(UNKNOWNWORD, next(noise_it)))
        for _ in range(noise_size)]


def add_unknown_word_to_utterances(utterances, replacement_string,
                                   unknown_word_prob, max_unknown_words,
                                   random_state):
    if not max_unknown_words:
        return utterances

    new_utterances = deepcopy(utterances)
    for u in new_utterances:
        if random_state.rand() < unknown_word_prob:
            num_unknown = random_state.randint(1, max_unknown_words + 1)
            # We choose to put the noise at the end of the sentence and not
            # in the middle so that it doesn't impact to much ngrams
            # computation
            extra_chunk = {
                TEXT: " " + " ".join(
                    replacement_string for _ in range(num_unknown))
            }
            u[DATA].append(extra_chunk)
    return new_utterances


def text_to_utterance(text):
    return {DATA: [{TEXT: text}]}
