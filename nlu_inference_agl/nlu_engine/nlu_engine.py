from __future__ import unicode_literals

import json
import logging
from builtins import str
from pathlib import Path

from future.utils import itervalues

from nlu_inference_agl.__about__ import __model_version__, __version__
from nlu_inference_agl.common.log_utils import log_elapsed_time
from nlu_inference_agl.common.utils import (fitted_required)
from nlu_inference_agl.constants import (
    AUTOMATICALLY_EXTENSIBLE, BUILTIN_ENTITY_PARSER, CUSTOM_ENTITY_PARSER,
    ENTITIES, ENTITY_KIND, LANGUAGE, RESOLVED_VALUE, RES_ENTITY,
    RES_INTENT, RES_INTENT_NAME, RES_MATCH_RANGE, RES_PROBA, RES_SLOTS,
    RES_VALUE, RESOURCES, BYPASS_VERSION_CHECK)
# from nlu_inference_agl.dataset import validate_and_format_dataset
from nlu_inference_agl.entity_parser import CustomEntityParser
from nlu_inference_agl.entity_parser.builtin_entity_parser import (
    BuiltinEntityParser, is_builtin_entity)
from nlu_inference_agl.exceptions import (
    InvalidInputError, IntentNotFoundError, LoadingError,
    IncompatibleModelError)
from nlu_inference_agl.intent_parser import IntentParser
from nlu_inference_agl.pipeline.configs import NLUEngineConfig
from nlu_inference_agl.pipeline.processing_unit import ProcessingUnit
from nlu_inference_agl.resources import load_resources_from_dir
from nlu_inference_agl.result import (
    builtin_slot, custom_slot, empty_result, extraction_result, is_empty,
    parsing_result)

logger = logging.getLogger(__name__)


@ProcessingUnit.register("nlu_engine")
class SnipsNLUEngine(ProcessingUnit):
    """Main class to use for intent parsing

    A :class:`SnipsNLUEngine` relies on a list of :class:`.IntentParser`
    object to parse intents, by calling them successively using the first
    positive output.

    With the default parameters, it will use the two following intent parsers
    in this order:

    - a :class:`.DeterministicIntentParser`
    - a :class:`.ProbabilisticIntentParser`

    The logic behind is to first use a conservative parser which has a very
    good precision while its recall is modest, so simple patterns will be
    caught, and then fallback on a second parser which is machine-learning
    based and will be able to parse unseen utterances while ensuring a good
    precision and recall.
    """

    config_type = NLUEngineConfig

    def __init__(self, config=None, **shared):
        """The NLU engine can be configured by passing a
        :class:`.NLUEngineConfig`"""
        super(SnipsNLUEngine, self).__init__(config, **shared)
        self.intent_parsers = []
        """list of :class:`.IntentParser`"""
        self.dataset_metadata = None

    @classmethod
    def default_config(cls):
        # Do not use the global default config, and use per-language default
        # configs instead
        return None
    
    @property
    def fitted(self):
        """Whether or not the nlu engine has already been fitted"""
        return self.dataset_metadata is not None

    @log_elapsed_time(logger, logging.DEBUG, "Parsed input in {elapsed_time}")
    @fitted_required
    def parse(self, text, intents=None, top_n=None):
        """Performs intent parsing on the provided *text* by calling its intent
        parsers successively

        Args:
            text (str): Input
            intents (str or list of str, optional): If provided, reduces the
                scope of intent parsing to the provided list of intents.
                The ``None`` intent is never filtered out, meaning that it can
                be returned even when using an intents scope.
            top_n (int, optional): when provided, this method will return a
                list of at most ``top_n`` most likely intents, instead of a
                single parsing result.
                Note that the returned list can contain less than ``top_n``
                elements, for instance when the parameter ``intents`` is not
                None, or when ``top_n`` is greater than the total number of
                intents.

        Returns:
            dict or list: the most likely intent(s) along with the extracted
            slots. See :func:`.parsing_result` and :func:`.extraction_result`
            for the output format.

        Raises:
            NotTrained: When the nlu engine is not fitted
            InvalidInputError: When input type is not unicode
        """
        if not isinstance(text, str):
            raise InvalidInputError("Expected unicode but received: %s"
                                    % type(text))

        if isinstance(intents, str):
            intents = {intents}
        elif isinstance(intents, list):
            intents = set(intents)

        if intents is not None:
            for intent in intents:
                if intent not in self.dataset_metadata["slot_name_mappings"]:
                    raise IntentNotFoundError(intent)

        if top_n is None:
            none_proba = 0.0
            for parser in self.intent_parsers:
                res = parser.parse(text, intents)
                if is_empty(res):
                    none_proba = res[RES_INTENT][RES_PROBA]
                    continue
                resolved_slots = self._resolve_slots(text, res[RES_SLOTS])
                return parsing_result(text, intent=res[RES_INTENT],
                                      slots=resolved_slots)
            return empty_result(text, none_proba)

        intents_results = self.get_intents(text)
        if intents is not None:
            intents_results = [res for res in intents_results
                               if res[RES_INTENT_NAME] is None
                               or res[RES_INTENT_NAME] in intents]
        intents_results = intents_results[:top_n]
        results = []
        for intent_res in intents_results:
            slots = self.get_slots(text, intent_res[RES_INTENT_NAME])
            results.append(extraction_result(intent_res, slots))
        return results

    @log_elapsed_time(logger, logging.DEBUG, "Got intents in {elapsed_time}")
    @fitted_required
    def get_intents(self, text):
        """Performs intent classification on the provided *text* and returns
        the list of intents ordered by decreasing probability

        The length of the returned list is exactly the number of intents in the
        dataset + 1 for the None intent

        .. note::

            The probabilities returned along with each intent are not
            guaranteed to sum to 1.0. They should be considered as scores
            between 0 and 1.
        """
        results = None
        for parser in self.intent_parsers:
            parser_results = parser.get_intents(text)
            if results is None:
                results = {res[RES_INTENT_NAME]: res for res in parser_results}
                continue

            for res in parser_results:
                intent = res[RES_INTENT_NAME]
                proba = max(res[RES_PROBA], results[intent][RES_PROBA])
                results[intent][RES_PROBA] = proba

        return sorted(itervalues(results), key=lambda res: -res[RES_PROBA])

    @log_elapsed_time(logger, logging.DEBUG, "Parsed slots in {elapsed_time}")
    @fitted_required
    def get_slots(self, text, intent):
        """Extracts slots from a text input, with the knowledge of the intent

        Args:
            text (str): input
            intent (str): the intent which the input corresponds to

        Returns:
            list: the list of extracted slots

        Raises:
            IntentNotFoundError: When the intent was not part of the training
                data
            InvalidInputError: When input type is not unicode
        """
        if not isinstance(text, str):
            raise InvalidInputError("Expected unicode but received: %s"
                                    % type(text))

        if intent is None:
            return []

        if intent not in self.dataset_metadata["slot_name_mappings"]:
            raise IntentNotFoundError(intent)

        for parser in self.intent_parsers:
            slots = parser.get_slots(text, intent)
            if not slots:
                continue
            return self._resolve_slots(text, slots)
        return []


    @classmethod
    def from_path(cls, path, **shared):
        """Loads a :class:`SnipsNLUEngine` instance from a directory path

        The data at the given path must have been generated using
        :func:`~SnipsNLUEngine.persist`

        Args:
            path (str): The path where the nlu engine is stored

        Raises:
            LoadingError: when some files are missing
            IncompatibleModelError: when trying to load an engine model which
                is not compatible with the current version of the lib
        """
        directory_path = Path(path)
        model_path = directory_path / "nlu_engine.json"
        if not model_path.exists():
            raise LoadingError("Missing nlu engine model file: %s"
                               % model_path.name)

        with model_path.open(encoding="utf8") as f:
            model = json.load(f)
        model_version = model.get("model_version")
        if model_version is None or model_version != __model_version__:
            bypass_version_check = shared.get(BYPASS_VERSION_CHECK, False)
            if bypass_version_check:
                logger.warning(
                    "Incompatible model version found. The library expected "
                    "'%s' but the loaded engine is '%s'. The NLU engine may "
                    "not load correctly.", __model_version__, model_version)
            else:
                raise IncompatibleModelError(model_version)

        dataset_metadata = model["dataset_metadata"]
        if shared.get(RESOURCES) is None and dataset_metadata is not None:
            language = dataset_metadata["language_code"]
            resources_dir = directory_path / "resources" / language
            if resources_dir.is_dir():
                resources = load_resources_from_dir(resources_dir)
                shared[RESOURCES] = resources

        if shared.get(BUILTIN_ENTITY_PARSER) is None:
            path = model["builtin_entity_parser"]
            if path is not None:
                parser_path = directory_path / path
                shared[BUILTIN_ENTITY_PARSER] = BuiltinEntityParser.from_path(
                    parser_path)

        if shared.get(CUSTOM_ENTITY_PARSER) is None:
            path = model["custom_entity_parser"]
            if path is not None:
                parser_path = directory_path / path
                shared[CUSTOM_ENTITY_PARSER] = CustomEntityParser.from_path(
                    parser_path)

        config = cls.config_type.from_dict(model["config"])
        nlu_engine = cls(config=config, **shared)
        nlu_engine.dataset_metadata = dataset_metadata
        intent_parsers = []
        for parser_idx, parser_name in enumerate(model["intent_parsers"]):
            parser_config = config.intent_parsers_configs[parser_idx]
            intent_parser_path = directory_path / parser_name
            intent_parser = IntentParser.load_from_path(
                intent_parser_path, parser_config.unit_name, **shared)
            intent_parsers.append(intent_parser)
        nlu_engine.intent_parsers = intent_parsers
        return nlu_engine

    def _resolve_slots(self, text, slots):
        builtin_scope = [slot[RES_ENTITY] for slot in slots
                         if is_builtin_entity(slot[RES_ENTITY])]
        custom_scope = [slot[RES_ENTITY] for slot in slots
                        if not is_builtin_entity(slot[RES_ENTITY])]
        # Do not use cached entities here as datetimes must be computed using
        # current context
        builtin_entities = self.builtin_entity_parser.parse(
            text, builtin_scope, use_cache=False)
        custom_entities = self.custom_entity_parser.parse(
            text, custom_scope, use_cache=True)

        resolved_slots = []
        for slot in slots:
            entity_name = slot[RES_ENTITY]
            raw_value = slot[RES_VALUE]
            is_builtin = is_builtin_entity(entity_name)
            if is_builtin:
                entities = builtin_entities
                parser = self.builtin_entity_parser
                slot_builder = builtin_slot
                use_cache = False
                extensible = False
            else:
                entities = custom_entities
                parser = self.custom_entity_parser
                slot_builder = custom_slot
                use_cache = True
                extensible = self.dataset_metadata[ENTITIES][entity_name][
                    AUTOMATICALLY_EXTENSIBLE]

            resolved_slot = None
            for ent in entities:
                if ent[ENTITY_KIND] == entity_name and \
                        ent[RES_MATCH_RANGE] == slot[RES_MATCH_RANGE]:
                    resolved_slot = slot_builder(slot, ent[RESOLVED_VALUE])
                    break
            if resolved_slot is None:
                matches = parser.parse(
                    raw_value, scope=[entity_name], use_cache=use_cache)
                if matches:
                    match = matches[0]
                    if is_builtin or len(match[RES_VALUE]) == len(raw_value):
                        resolved_slot = slot_builder(
                            slot, match[RESOLVED_VALUE])

            if resolved_slot is None and extensible:
                resolved_slot = slot_builder(slot)

            if resolved_slot is not None:
                resolved_slots.append(resolved_slot)

        return resolved_slots