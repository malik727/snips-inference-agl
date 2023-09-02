from __future__ import division, unicode_literals

import json
from builtins import str, zip
from copy import deepcopy
from pathlib import Path

from future.utils import iteritems

from snips_inference_agl.common.utils import (
    fitted_required, replace_entities_with_placeholders)
from snips_inference_agl.constants import (
    DATA, ENTITY, ENTITY_KIND, NGRAM, TEXT)
from snips_inference_agl.dataset import get_text_from_chunks
from snips_inference_agl.entity_parser.builtin_entity_parser import (
    is_builtin_entity)
from snips_inference_agl.exceptions import (LoadingError)
from snips_inference_agl.languages import get_default_sep
from snips_inference_agl.pipeline.configs import FeaturizerConfig
from snips_inference_agl.pipeline.configs.intent_classifier import (
    CooccurrenceVectorizerConfig, TfidfVectorizerConfig)
from snips_inference_agl.pipeline.processing_unit import ProcessingUnit
from snips_inference_agl.preprocessing import stem, tokenize_light
from snips_inference_agl.resources import get_stop_words, get_word_cluster
from snips_inference_agl.slot_filler.features_utils import get_all_ngrams


@ProcessingUnit.register("featurizer")
class Featurizer(ProcessingUnit):
    """Feature extractor for text classification relying on ngrams tfidf and
    optionally word cooccurrences features"""

    config_type = FeaturizerConfig

    def __init__(self, config=None, **shared):
        super(Featurizer, self).__init__(config, **shared)
        self.language = None
        self.tfidf_vectorizer = None
        self.cooccurrence_vectorizer = None

    @property
    def fitted(self):
        if not self.tfidf_vectorizer or not self.tfidf_vectorizer.vocabulary:
            return False
        return True
    
    def transform(self, utterances):
        import scipy.sparse as sp

        x = self.tfidf_vectorizer.transform(utterances)
        if self.cooccurrence_vectorizer:
            x_cooccurrence = self.cooccurrence_vectorizer.transform(utterances)
            x = sp.hstack((x, x_cooccurrence))
        return x

    @classmethod
    def from_path(cls, path, **shared):
        path = Path(path)

        model_path = path / "featurizer.json"
        if not model_path.exists():
            raise LoadingError("Missing featurizer model file: %s"
                               % model_path.name)
        with model_path.open("r", encoding="utf-8") as f:
            featurizer_dict = json.load(f)

        featurizer_config = featurizer_dict["config"]
        featurizer = cls(featurizer_config, **shared)

        featurizer.language = featurizer_dict["language_code"]

        tfidf_vectorizer = featurizer_dict["tfidf_vectorizer"]
        if tfidf_vectorizer:
            vectorizer_path = path / featurizer_dict["tfidf_vectorizer"]
            tfidf_vectorizer = TfidfVectorizer.from_path(
                vectorizer_path, **shared)
        featurizer.tfidf_vectorizer = tfidf_vectorizer

        cooccurrence_vectorizer = featurizer_dict["cooccurrence_vectorizer"]
        if cooccurrence_vectorizer:
            vectorizer_path = path / featurizer_dict["cooccurrence_vectorizer"]
            cooccurrence_vectorizer = CooccurrenceVectorizer.from_path(
                vectorizer_path, **shared)
        featurizer.cooccurrence_vectorizer = cooccurrence_vectorizer

        return featurizer


@ProcessingUnit.register("tfidf_vectorizer")
class TfidfVectorizer(ProcessingUnit):
    """Wrapper of the scikit-learn TfidfVectorizer"""

    config_type = TfidfVectorizerConfig

    def __init__(self, config=None, **shared):
        super(TfidfVectorizer, self).__init__(config, **shared)
        self._tfidf_vectorizer = None
        self._language = None
        self.builtin_entity_scope = None

    @property
    def fitted(self):
        return self._tfidf_vectorizer is not None and hasattr(
            self._tfidf_vectorizer, "vocabulary_")
    
    @fitted_required
    def transform(self, x):
        """Featurizes the given utterances after enriching them with builtin
        entities matches, custom entities matches and the potential word
        clusters matches

        Args:
            x (list of dict): list of utterances

        Returns:
            :class:`.scipy.sparse.csr_matrix`: A sparse matrix X of shape
            (len(x), len(self.vocabulary)) where X[i, j] contains tfdif of
            the ngram of index j of the vocabulary in the utterance i

        Raises:
            NotTrained: when the vectorizer is not fitted:
        """
        utterances = [self._enrich_utterance(*data)
                      for data in zip(*self._preprocess(x))]
        return self._tfidf_vectorizer.transform(utterances)

    def _preprocess(self, utterances):
        normalized_utterances = deepcopy(utterances)
        for u in normalized_utterances:
            nb_chunks = len(u[DATA])
            for i, chunk in enumerate(u[DATA]):
                chunk[TEXT] = _normalize_stem(
                    chunk[TEXT], self.language, self.resources,
                    self.config.use_stemming)
                if i < nb_chunks - 1:
                    chunk[TEXT] += " "

        # Extract builtin entities on unormalized utterances
        builtin_ents = [
            self.builtin_entity_parser.parse(
                get_text_from_chunks(u[DATA]),
                self.builtin_entity_scope, use_cache=True)
            for u in utterances
        ]
        # Extract builtin entities on normalized utterances
        custom_ents = [
            self.custom_entity_parser.parse(
                get_text_from_chunks(u[DATA]), use_cache=True)
            for u in normalized_utterances
        ]
        if self.config.word_clusters_name:
            # Extract world clusters on unormalized utterances
            original_utterances_text = [get_text_from_chunks(u[DATA])
                                        for u in utterances]
            w_clusters = [
                _get_word_cluster_features(
                    tokenize_light(u.lower(), self.language),
                    self.config.word_clusters_name,
                    self.resources)
                for u in original_utterances_text
            ]
        else:
            w_clusters = [None for _ in normalized_utterances]

        return normalized_utterances, builtin_ents, custom_ents, w_clusters

    def _enrich_utterance(self, utterance, builtin_entities, custom_entities,
                          word_clusters):
        custom_entities_features = [
            _entity_name_to_feature(e[ENTITY_KIND], self.language)
            for e in custom_entities]

        builtin_entities_features = [
            _builtin_entity_to_feature(ent[ENTITY_KIND], self.language)
            for ent in builtin_entities
        ]

        # We remove values of builtin slots from the utterance to avoid
        # learning specific samples such as '42' or 'tomorrow'
        filtered_tokens = [
            chunk[TEXT] for chunk in utterance[DATA]
            if ENTITY not in chunk or not is_builtin_entity(chunk[ENTITY])
        ]

        features = get_default_sep(self.language).join(filtered_tokens)

        if builtin_entities_features:
            features += " " + " ".join(sorted(builtin_entities_features))
        if custom_entities_features:
            features += " " + " ".join(sorted(custom_entities_features))
        if word_clusters:
            features += " " + " ".join(sorted(word_clusters))

        return features

    @property
    def language(self):
        # Create this getter to prevent the language from being set elsewhere
        # than in the fit
        return self._language

    @property
    def vocabulary(self):
        if self._tfidf_vectorizer and hasattr(
                self._tfidf_vectorizer, "vocabulary_"):
            return self._tfidf_vectorizer.vocabulary_
        return None

    @property
    def idf_diag(self):
        if self._tfidf_vectorizer and hasattr(
                self._tfidf_vectorizer, "vocabulary_"):
            return self._tfidf_vectorizer.idf_
        return None

    @classmethod
    # pylint: disable=W0212
    def from_path(cls, path, **shared):
        import numpy as np
        import scipy.sparse as sp
        from sklearn.feature_extraction.text import (
            TfidfTransformer, TfidfVectorizer as SklearnTfidfVectorizer)

        path = Path(path)

        model_path = path / "vectorizer.json"
        if not model_path.exists():
            raise LoadingError("Missing vectorizer model file: %s"
                               % model_path.name)
        with model_path.open("r", encoding="utf-8") as f:
            vectorizer_dict = json.load(f)

        vectorizer = cls(vectorizer_dict["config"], **shared)
        vectorizer._language = vectorizer_dict["language_code"]

        builtin_entity_scope = vectorizer_dict["builtin_entity_scope"]
        if builtin_entity_scope is not None:
            builtin_entity_scope = set(builtin_entity_scope)
        vectorizer.builtin_entity_scope = builtin_entity_scope

        vectorizer_ = vectorizer_dict["vectorizer"]
        if vectorizer_:
            vocab = vectorizer_["vocab"]
            idf_diag_data = vectorizer_["idf_diag"]
            idf_diag_data = np.array(idf_diag_data)

            idf_diag_shape = (len(idf_diag_data), len(idf_diag_data))
            row = list(range(idf_diag_shape[0]))
            col = list(range(idf_diag_shape[0]))
            idf_diag = sp.csr_matrix(
                (idf_diag_data, (row, col)), shape=idf_diag_shape)

            tfidf_transformer = TfidfTransformer()
            tfidf_transformer._idf_diag = idf_diag

            vectorizer_ = SklearnTfidfVectorizer(
                tokenizer=lambda x: tokenize_light(x, vectorizer._language))
            vectorizer_.vocabulary_ = vocab

            vectorizer_._tfidf = tfidf_transformer

        vectorizer._tfidf_vectorizer = vectorizer_
        return vectorizer


@ProcessingUnit.register("cooccurrence_vectorizer")
class CooccurrenceVectorizer(ProcessingUnit):
    """Featurizer that takes utterances and extracts ordered word cooccurrence
     features matrix from them"""

    config_type = CooccurrenceVectorizerConfig

    def __init__(self, config=None, **shared):
        super(CooccurrenceVectorizer, self).__init__(config, **shared)
        self._word_pairs = None
        self._language = None
        self.builtin_entity_scope = None

    @property
    def language(self):
        # Create this getter to prevent the language from being set elsewhere
        # than in the fit
        return self._language

    @property
    def word_pairs(self):
        return self._word_pairs

    @property
    def fitted(self):
        """Whether or not the vectorizer is fitted"""
        return self.word_pairs is not None
    
    @fitted_required
    def transform(self, x):
        """Computes the cooccurrence feature matrix.

        Args:
            x (list of dict): list of utterances

        Returns:
            :class:`.scipy.sparse.csr_matrix`: A sparse matrix X of shape
            (len(x), len(self.word_pairs)) where X[i, j] = 1.0 if
            x[i][0] contains the words cooccurrence (w1, w2) and if
            self.word_pairs[(w1, w2)] = j

        Raises:
            NotTrained: when the vectorizer is not fitted
        """
        import numpy as np
        import scipy.sparse as sp

        preprocessed = self._preprocess(x)
        utterances = [
            self._enrich_utterance(utterance, builtin_ents, custom_ent)
            for utterance, builtin_ents, custom_ent in zip(*preprocessed)]

        x_coo = sp.dok_matrix((len(x), len(self.word_pairs)), dtype=np.int32)
        for i, u in enumerate(utterances):
            for p in self._extract_word_pairs(u):
                if p in self.word_pairs:
                    x_coo[i, self.word_pairs[p]] = 1

        return x_coo.tocsr()

    def _preprocess(self, x):
        # Extract all entities on unnormalized data
        builtin_ents = [
            self.builtin_entity_parser.parse(
                get_text_from_chunks(u[DATA]),
                self.builtin_entity_scope,
                use_cache=True
            ) for u in x
        ]
        custom_ents = [
            self.custom_entity_parser.parse(
                get_text_from_chunks(u[DATA]), use_cache=True)
            for u in x
        ]
        return x, builtin_ents, custom_ents

    def _extract_word_pairs(self, utterance):
        if self.config.filter_stop_words:
            stop_words = get_stop_words(self.resources)
            utterance = [t for t in utterance if t not in stop_words]
        pairs = set()
        for j, w1 in enumerate(utterance):
            max_index = None
            if self.config.window_size is not None:
                max_index = j + self.config.window_size + 1
            for w2 in utterance[j + 1:max_index]:
                key = (w1, w2)
                if not self.config.keep_order:
                    key = tuple(sorted(key))
                pairs.add(key)
        return pairs
    
    def _enrich_utterance(self, x, builtin_ents, custom_ents):
        utterance = get_text_from_chunks(x[DATA])
        all_entities = builtin_ents + custom_ents
        placeholder_fn = self._placeholder_fn
        # Replace entities with placeholders
        enriched_utterance = replace_entities_with_placeholders(
            utterance, all_entities, placeholder_fn)[1]
        # Tokenize
        enriched_utterance = tokenize_light(enriched_utterance, self.language)
        # Remove the unknownword strings if needed
        if self.config.unknown_words_replacement_string:
            enriched_utterance = [
                t for t in enriched_utterance
                if t != self.config.unknown_words_replacement_string
            ]
        return enriched_utterance

    def _extract_word_pairs(self, utterance):
        if self.config.filter_stop_words:
            stop_words = get_stop_words(self.resources)
            utterance = [t for t in utterance if t not in stop_words]
        pairs = set()
        for j, w1 in enumerate(utterance):
            max_index = None
            if self.config.window_size is not None:
                max_index = j + self.config.window_size + 1
            for w2 in utterance[j + 1:max_index]:
                key = (w1, w2)
                if not self.config.keep_order:
                    key = tuple(sorted(key))
                pairs.add(key)
        return pairs

    def _placeholder_fn(self, entity_name):
        return "".join(
            tokenize_light(str(entity_name), str(self.language))).upper()

    @classmethod
    # pylint: disable=protected-access
    def from_path(cls, path, **shared):
        path = Path(path)
        model_path = path / "vectorizer.json"
        if not model_path.exists():
            raise LoadingError("Missing vectorizer model file: %s"
                               % model_path.name)

        with model_path.open(encoding="utf8") as f:
            vectorizer_dict = json.load(f)
        config = vectorizer_dict.pop("config")

        self = cls(config, **shared)
        self._language = vectorizer_dict["language_code"]
        self._word_pairs = None

        builtin_entity_scope = vectorizer_dict["builtin_entity_scope"]
        if builtin_entity_scope is not None:
            builtin_entity_scope = set(builtin_entity_scope)
        self.builtin_entity_scope = builtin_entity_scope

        if vectorizer_dict["word_pairs"]:
            self._word_pairs = {
                tuple(p): int(i)
                for i, p in iteritems(vectorizer_dict["word_pairs"])
            }
        return self

def _entity_name_to_feature(entity_name, language):
    return "entityfeature%s" % "".join(tokenize_light(
        entity_name.lower(), language))


def _builtin_entity_to_feature(builtin_entity_label, language):
    return "builtinentityfeature%s" % "".join(tokenize_light(
        builtin_entity_label.lower(), language))


def _normalize_stem(text, language, resources, use_stemming):
    from snips_nlu_utils import normalize

    if use_stemming:
        return stem(text, language, resources)
    return normalize(text)


def _get_word_cluster_features(query_tokens, clusters_name, resources):
    if not clusters_name:
        return []
    ngrams = get_all_ngrams(query_tokens)
    cluster_features = []
    for ngram in ngrams:
        cluster = get_word_cluster(resources, clusters_name).get(
            ngram[NGRAM].lower(), None)
        if cluster is not None:
            cluster_features.append(cluster)
    return cluster_features
