# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
import logging
from typing import Optional, Union

import torch

from hyperbox.mutables import spaces
from hyperbox.mutator.base_mutator import BaseMutator
from hyperbox.utils.utils import TorchTensorEncoder, lazy_property

logger = logging.getLogger(__name__)


class Mutator(BaseMutator):

    def __init__(self, model):
        super().__init__(model)
        self._cache = dict()
        default_mask = dict()
        for m in model.modules():
            if isinstance(m, spaces.Mutable):
                default_mask[m.key] = m.mask
        self.default_mask = default_mask

    @lazy_property
    def has_duplicate_mutable(self):
        '''check if the model has duplicate mutable
            e.g., in DARTS, different cells share the same mutable as the edges use the same key,
            but only the mask of the last cell will be updated by `reset`.
            To avoid this case, you should use `sync_mask_to_duplicate_mutables` to sync the mask to the duplicate mutables.
            more details can be found in `reset`.
        '''
        mutable_keys = set()
        for name, module in self.model.named_modules():
            if isinstance(module, spaces.Mutable):
                if module.key not in mutable_keys:
                    mutable_keys.add(module.key)
                else:
                    return True
        return False

    def build_archs_for_valid(self, *args, **kwargs):
        '''
        Build a list of archs for validation

        Returns
        -------
        a list of dict
        '''
        if hasattr(self.model, 'build_archs_for_valid'):
            archs_to_valid = self.model.build_archs_for_valid(*args, **kwargs)
        else:
            archs_to_valid = [self.default_mask]
        self.archs_to_valid = archs_to_valid

    def sample_search(self, *args, **kwargs):
        """
        Override to implement this method to iterate over mutables and make decisions.

        Returns
        -------
        dict
            A mapping from key of mutables to decisions.
        """
        raise NotImplementedError

    def sample_final(self, *args, **kwargs):
        """
        Override to implement this method to iterate over mutables and make decisions that is final
        for export and retraining.

        Returns
        -------
        dict
            A mapping from key of mutables to decisions.
        """
        raise NotImplementedError

    def sample_by_mask(self, mask: dict):
        '''
        Sample an architecture by the mask
        '''
        self._cache = self.check_freeze_mutable(mask)
        for mutable in self.mutables:
            assert mutable.mask.shape==mask[mutable.key].shape,\
                f"the given mask ({mask[mutable.key].shape}) cannot match the original size [{mutable.key}]{mutable.mask.shape}"
            mutable.mask = mask[mutable.key]
        if self.has_duplicate_mutable:
            self.sync_mask_to_duplicate_mutables(self._cache)

    def reset(self, *args, **kwargs):
        """
        Reset the mutator by call the `sample_search` to resample (for search). Stores the result in a local
        variable so that `on_forward_operation_space` and `on_forward_input_space` can use the decision directly.

        Returns
        -------
        None
        """
        if not hasattr(self, 'sample_func'):
            self._cache = self.sample_search(*args, **kwargs)
        else:
            self._cache = self.sample_func(self, *args, **kwargs)
            del self.sample_func
        self._cache = self.check_freeze_mutable(self._cache)
        if self.has_duplicate_mutable:
            self.sync_mask_to_duplicate_mutables(self._cache)
        return self._cache

    def check_freeze_mutable(self, mask):
        '''
        Check if the mutable is frozen, if so, set the mask to be the default mask
        '''
        for mutable in self.mutables:
            if getattr(mutable, 'is_freeze', False):
                mask[mutable.key] = mutable.mask
        return mask

    def sync_mask_to_duplicate_mutables(self, mask):
        '''
        Sync the mask to the duplicate mutables
        '''
        for name, module in self.model.named_modules():
            if isinstance(module, spaces.Mutable):
                if module.key in mask:
                # if module.key in mask and not all(mask[module.key] == module.mask):
                    module.mask.data = mask[module.key].data

    def export(self, *args, **kwargs):
        """
        Resample (for final) and return results.

        Returns
        -------
        dict
        """
        return self.sample_final(*args, **kwargs)

    def save_arch(self, file_path):
        mask = self._cache
        with open(file_path, "w") as f:
            json.dump(mask, f, indent=4, sort_keys=True, cls=TorchTensorEncoder)

    def on_forward_operation_space(self, mutable, *inputs):
        """
        On default, this method calls :meth:`on_calc_layer_choice_mask` to get a mask on how to choose between layers
        (either by switch or by weights), then it will reduce the list of all tensor outputs with the policy specified
        in `mutable.reduction`. It will also cache the mask with corresponding `mutable.key`.

        Parameters
        ----------
        mutable : OperationSpace
        inputs : list of torch.Tensor

        Returns
        -------
        tuple of torch.Tensor and torch.Tensor
        """

        def _map_fn(op, *inputs):
            return op(*inputs)

        mask = self._get_decision(mutable) # 从mutable中获取建议，比如随机采样
        assert len(mask) == len(mutable.candidates), \
            "Invalid mask, expected {} to be of length {}.".format(mask, len(mutable.candidates))
        out = self._select_with_mask(_map_fn, [(choice, *inputs) for choice in mutable.candidates], mask)
        return self._tensor_reduction(mutable.reduction, out), mask

    def on_forward_input_space(self, mutable, tensor_list):
        """
        On default, this method calls :meth:`on_calc_input_choice_mask` with `tags`
        to get a mask on how to choose between inputs (either by switch or by weights), then it will reduce
        the list of all tensor outputs with the policy specified in `mutable.reduction`. It will also cache the
        mask with corresponding `mutable.key`.

        Parameters
        ----------
        mutable : InputSpace
        tensor_list : list of torch.Tensor
        tags : list of string

        Returns
        -------
        tuple of torch.Tensor and torch.Tensor
        """
        mask = self._get_decision(mutable)
        assert len(mask) == mutable.n_candidates, \
            "Invalid mask, expected {} to be of length {}.".format(mask, mutable.n_candidates)
        out = self._select_with_mask(lambda x: x, [(t,) for t in tensor_list], mask)
        return self._tensor_reduction(mutable.reduction, out), mask

    def _select_with_mask(self, map_fn, candidates, mask):
        if "BoolTensor" in mask.type():
            out = [map_fn(*cand) for cand, m in zip(candidates, mask) if m]
        elif "FloatTensor" in mask.type():
            out = [map_fn(*cand) * m for cand, m in zip(candidates, mask) if m]
        else:
            raise ValueError("Unrecognized mask")
        return out

    def _tensor_reduction(self, reduction_type, tensor_list):
        if reduction_type == "none":
            return tensor_list
        if not tensor_list:
            return None  # empty. return None for now
        if len(tensor_list) == 1:
            return tensor_list[0]
        if reduction_type == "sum":
            return sum(tensor_list)
        if reduction_type == "mean":
            return sum(tensor_list) / len(tensor_list)
        if reduction_type == "concat":
            return torch.cat(tensor_list, dim=1)
        raise ValueError("Unrecognized reduction policy: \"{}\"".format(reduction_type))

    def _get_decision(self, mutable):
        """
        By default, this method checks whether `mutable.key` is already in the decision cache,
        and returns the result without double-check.

        Parameters
        ----------
        mutable : Mutable

        Returns
        -------
        object
        """
        if mutable.key not in self._cache:
            raise ValueError("\"{}\" not found in decision cache.".format(mutable.key))
        result = self._cache[mutable.key]
        logger.debug("Decision %s: %s", mutable.key, result)
        return result

    def get_mutable_by_key(self, key):
        if len(self._cache) > 0:
            return self._cache[key]
        else:
            for mutable in self.mutables:
                if mutable.key == key:
                    return mutable
        raise ValueError("Mutable with key \"{}\" not found.".format(key))

    def __getitem__(self, key: Optional[Union[str, int]]) -> 'spaces.Mutable':
        _m = None
        if isinstance(key, int):
            for idx, mutable in enumerate(self.mutables):
                if idx == key:
                    _m = mutable
        elif isinstance(key, str):
            _m = self.get_mutable_by_key(key)
        if _m is None:
            raise ValueError("Invalid key: {}".format(key))
        return _m

    def __len__(self):
        return self.num_mutables

    @lazy_property
    def num_mutables(self):
        num = 0
        for mutable in self.mutables:
            num += 1
        return num
