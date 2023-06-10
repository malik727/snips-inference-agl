from __future__ import unicode_literals

import json
import logging
from builtins import range, str, zip
from pathlib import Path

from nlu_inference_agl.common.log_utils import DifferedLoggingMessage, log_elapsed_time
from nlu_inference_agl.common.utils import (
    check_persisted_path, fitted_required, json_string)
from nlu_inference_agl.constants import LANGUAGE, RES_PROBA
from nlu_inference_agl.dataset import validate_and_format_dataset
from nlu_inference_agl.exceptions import LoadingError
from nlu_inference_agl.intent_classifier.featurizer import Featurizer
from nlu_inference_agl.intent_classifier.intent_classifier import IntentClassifier
from nlu_inference_agl.intent_classifier.log_reg_classifier_utils import (
    build_training_data, get_regularization_factor, text_to_utterance)
from nlu_inference_agl.pipeline.configs import LogRegIntentClassifierConfig
from nlu_inference_agl.result import intent_classification_result

logger = logging.getLogger(__name__)

# We set tol to 1e-3 to silence the following warning with Python 2 (
# scikit-learn 0.20):
#
# FutureWarning: max_iter and tol parameters have been added in SGDClassifier
# in 0.19. If max_iter is set but tol is left unset, the default value for tol
# in 0.19 and 0.20 will be None (which is equivalent to -infinity, so it has no
# effect) but will change in 0.21 to 1e-3. Specify tol to silence this warning.

LOG_REG_ARGS = {
    "loss": "log",
    "penalty": "l2",
    "max_iter": 1000,
    "tol": 1e-3,
    "n_jobs": -1
}


@IntentClassifier.register("log_reg_intent_classifier")
class LogRegIntentClassifier(IntentClassifier):
    """Intent classifier which uses a Logistic Regression underneath"""

    config_type = LogRegIntentClassifierConfig

    def __init__(self, config=None, **shared):
        """The LogReg intent classifier can be configured by passing a
        :class:`.LogRegIntentClassifierConfig`"""
        super(LogRegIntentClassifier, self).__init__(config, **shared)
        self.classifier = None
        self.intent_list = None
        self.featurizer = None

    @property
    def fitted(self):
        """Whether or not the intent classifier has already been fitted"""
        return self.intent_list is not None

    @fitted_required
    def get_intent(self, text, intents_filter=None):
        """Performs intent classification on the provided *text*

        Args:
            text (str): Input
            intents_filter (str or list of str): When defined, it will find
                the most likely intent among the list, otherwise it will use
                the whole list of intents defined in the dataset

        Returns:
            dict or None: The most likely intent along with its probability or
            *None* if no intent was found

        Raises:
            :class:`snips_nlu.exceptions.NotTrained`: When the intent
                classifier is not fitted

        """
        return self._get_intents(text, intents_filter)[0]

    @fitted_required
    def get_intents(self, text):
        """Performs intent classification on the provided *text* and returns
        the list of intents ordered by decreasing probability

        The length of the returned list is exactly the number of intents in the
        dataset + 1 for the None intent

        Raises:
            :class:`snips_nlu.exceptions.NotTrained`: when the intent
                classifier is not fitted
        """
        return self._get_intents(text, intents_filter=None)

    def _get_intents(self, text, intents_filter):
        if isinstance(intents_filter, str):
            intents_filter = {intents_filter}
        elif isinstance(intents_filter, list):
            intents_filter = set(intents_filter)

        if not text or not self.intent_list or not self.featurizer:
            results = [intent_classification_result(None, 1.0)]
            results += [intent_classification_result(i, 0.0)
                        for i in self.intent_list if i is not None]
            return results

        if len(self.intent_list) == 1:
            return [intent_classification_result(self.intent_list[0], 1.0)]

        # pylint: disable=C0103
        X = self.featurizer.transform([text_to_utterance(text)])
        # pylint: enable=C0103
        proba_vec = self._predict_proba(X)
        logger.debug(
            "%s", DifferedLoggingMessage(self.log_activation_weights, text, X))
        results = [
            intent_classification_result(i, proba)
            for i, proba in zip(self.intent_list, proba_vec[0])
            if intents_filter is None or i is None or i in intents_filter]

        return sorted(results, key=lambda res: -res[RES_PROBA])

    def _predict_proba(self, X):  # pylint: disable=C0103
        import numpy as np

        self.classifier._check_proba()  # pylint: disable=W0212

        prob = self.classifier.decision_function(X)
        prob *= -1
        np.exp(prob, prob)
        prob += 1
        np.reciprocal(prob, prob)
        if prob.ndim == 1:
            return np.vstack([1 - prob, prob]).T
        return prob

    @classmethod
    def from_path(cls, path, **shared):
        """Loads a :class:`LogRegIntentClassifier` instance from a path

        The data at the given path must have been generated using
        :func:`~LogRegIntentClassifier.persist`
        """
        import numpy as np
        from sklearn.linear_model import SGDClassifier

        path = Path(path)
        model_path = path / "intent_classifier.json"
        if not model_path.exists():
            raise LoadingError("Missing intent classifier model file: %s"
                               % model_path.name)

        with model_path.open(encoding="utf8") as f:
            model_dict = json.load(f)

        # Create the classifier
        config = LogRegIntentClassifierConfig.from_dict(model_dict["config"])
        intent_classifier = cls(config=config, **shared)
        intent_classifier.intent_list = model_dict['intent_list']

        # Create the underlying SGD classifier
        sgd_classifier = None
        coeffs = model_dict['coeffs']
        intercept = model_dict['intercept']
        t_ = model_dict["t_"]
        if coeffs is not None and intercept is not None:
            sgd_classifier = SGDClassifier(**LOG_REG_ARGS)
            sgd_classifier.coef_ = np.array(coeffs)
            sgd_classifier.intercept_ = np.array(intercept)
            sgd_classifier.t_ = t_
        intent_classifier.classifier = sgd_classifier

        # Add the featurizer
        featurizer = model_dict['featurizer']
        if featurizer is not None:
            featurizer_path = path / featurizer
            intent_classifier.featurizer = Featurizer.from_path(
                featurizer_path, **shared)

        return intent_classifier
    
    def log_activation_weights(self, text, x, top_n=50):
        import numpy as np

        if not hasattr(self.featurizer, "feature_index_to_feature_name"):
            return None

        log = "\n\nTop {} feature activations for: \"{}\":\n".format(
            top_n, text)
        activations = np.multiply(
            self.classifier.coef_, np.asarray(x.todense()))
        abs_activation = np.absolute(activations).flatten().squeeze()

        if top_n > activations.size:
            top_n = activations.size

        top_n_activations_ix = np.argpartition(abs_activation, -top_n,
                                               axis=None)[-top_n:]
        top_n_activations_ix = np.unravel_index(
            top_n_activations_ix, activations.shape)

        index_to_feature = self.featurizer.feature_index_to_feature_name
        features_intent_and_activation = [
            (self.intent_list[i], index_to_feature[f], activations[i, f])
            for i, f in zip(*top_n_activations_ix)]

        features_intent_and_activation = sorted(
            features_intent_and_activation, key=lambda x: abs(x[2]),
            reverse=True)

        for intent, feature, activation in features_intent_and_activation:
            log += "\n\n\"{}\" -> ({}, {:.2f})".format(
                intent, feature, float(activation))
        log += "\n\n"
        return log
