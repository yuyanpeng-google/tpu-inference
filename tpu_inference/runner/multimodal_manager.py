from typing import TYPE_CHECKING

import numpy as np

import jax
import jax.numpy as jnp
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.multimodal.inputs import MultiModalKwargsItem, PlaceholderRange
from vllm.multimodal.utils import group_mm_kwargs_by_modality
from vllm.v1.core.sched.output import SchedulerOutput as VllmSchedulerOutput
from vllm.v1.worker.utils import (gather_mm_placeholders,
                                  scatter_mm_placeholders)

from tpu_inference.models.jax.utils.multi_modal_utils import (
    flatten_embeddings, sanity_check_mm_encoder_outputs)


import os
DEV_NEW_MM = True if os.environ.get("DEV_NEW_MM", None) is not None else False

from tpu_inference.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from tpu_inference.runner.tpu_runner import TPUModelRunner

def scatter_mm_placeholders_jax(
    embeds: jax.Array,
    is_embed: np.ndarray | None,
) -> jax.Array:
    """
    TODO: MODIFY DOC STRING
    Scatter the multimodal embeddings into a contiguous tensor that represents
    the placeholder tokens.

    [`vllm.multimodal.processing.PromptUpdateDetails.is_embed`][].

    Args:
        embeds: The multimodal embeddings.
            Shape: `(num_embeds, embed_dim)`
        is_embed: A boolean mask indicating which positions in the placeholder
            tokens need to be filled with multimodal embeddings.
            Shape: `(num_placeholders, num_embeds)`
    """
    if is_embed is None:
        return embeds

    # placeholders = embeds.new_full(
    #     (is_embed.shape[0], embeds.shape[-1]),
    #     fill_value=torch.nan,
    # )
    placeholders = jnp.full(
        (is_embed.shape[0], embeds.shape[-1]),
        fill_value=jnp.nan,
        dtype=embeds.dtype,
        device=embeds.device,
    )
    placeholders = placeholders.at[is_embed].set(embeds)
    return placeholders

class MultiModalManager:

    def __init__(self, runner: "TPUModelRunner"):
        self.runner = runner

    def calc_mrope_positions(self, scheduler_output: "VllmSchedulerOutput"):
        mrope_pos_ptr = 0
        for index, req_id in enumerate(self.runner.input_batch.req_ids):
            req = self.runner.requests[req_id]
            assert req.mrope_positions is not None

            num_computed_tokens = \
                self.runner.input_batch.num_computed_tokens_cpu[index]
            num_scheduled_tokens = \
                scheduler_output.num_scheduled_tokens[req_id]
            num_prompt_tokens = len(req.prompt_token_ids)

            if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:
                prompt_part_len = max(0,
                                      num_prompt_tokens - num_computed_tokens)
                completion_part_len = max(
                    0, num_scheduled_tokens - prompt_part_len)
            else:
                prompt_part_len = num_scheduled_tokens
                completion_part_len = 0

            assert num_scheduled_tokens == prompt_part_len + completion_part_len

            if prompt_part_len > 0:
                # prompt's mrope_positions are pre-computed
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + prompt_part_len
                src_start = num_computed_tokens
                src_end = num_computed_tokens + prompt_part_len

                self.runner.mrope_positions_cpu[:, dst_start:dst_end] = \
                    req.mrope_positions[:,src_start:src_end]

                mrope_pos_ptr += prompt_part_len

            if completion_part_len > 0:
                # compute completion's mrope_positions on-the-fly
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + completion_part_len

                MRotaryEmbedding.get_next_input_positions_tensor(
                    out=self.runner.mrope_positions_cpu,
                    out_offset=dst_start,
                    mrope_position_delta=req.mrope_position_delta,
                    context_len=num_computed_tokens + prompt_part_len,
                    num_new_tokens=completion_part_len,
                )

                mrope_pos_ptr += completion_part_len

    def execute_mm_encoder_dev(self, scheduler_output: "VllmSchedulerOutput"):
        import torch
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return

        # Batch the multi-modal inputs.
        mm_kwargs = list[MultiModalKwargsItem]()
        # List of tuple (mm_hash, pos_info)
        mm_hashes_pos = list[tuple[str, PlaceholderRange]]()
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.runner.requests[req_id]
            for mm_input_id in encoder_input_ids:
                mm_feature = req_state.mm_features[mm_input_id]
                mm_hash = mm_feature.identifier
                mm_kwargs.append(mm_feature.data)
                mm_hashes_pos.append((mm_hash, mm_feature.mm_position))

        # Batch mm inputs as much as we can: if a request in the batch has
        # multiple modalities or a different modality than the previous one,
        # we process it separately to preserve item order.
        # FIXME(ywang96): This is a hacky way to deal with multiple modalities
        # in the same batch while still being able to benefit from batching
        # multimodal inputs. The proper solution should be reordering the
        # encoder outputs.
        encoder_outputs = []
        for _, num_items, mm_kwargs_group in group_mm_kwargs_by_modality(
                mm_kwargs, merge_by_field_config=self.runner.model.model.vllm_model.merge_by_field_config):
            batched_mm_inputs = mm_kwargs_group
            # Convert torch tensors to numpy arrays that JAX can handle.
            if "pixel_values" in batched_mm_inputs and isinstance(
                    batched_mm_inputs["pixel_values"], list):
                batched_mm_inputs["pixel_values"] = torch.cat(
                    batched_mm_inputs["pixel_values"], dim=0)

            image_grid_thw = ()
            for key, value in batched_mm_inputs.items():
                if isinstance(value, torch.Tensor):
                    if key == 'image_grid_thw':
                        # change it to tuple of tuples to make it hashable for JIT

                        # Shape: (B, N, 3) -> (B*N, 3) -> tuple of tuples
                        grid_thw_tensor = batched_mm_inputs[key]
                        grid_thw_reshaped = grid_thw_tensor.reshape(-1, 3)
                        image_grid_thw = tuple(
                            tuple(row) for row in grid_thw_reshaped.tolist())

                        continue

                    if value.dtype == torch.bfloat16 and False:
                        batched_mm_inputs[key] = value.to(
                            torch.float32).numpy().astype(jnp.bfloat16)
                    else:
                        # batched_mm_inputs[key] = value.numpy()
                        batched_mm_inputs[key] = value

            if batched_mm_inputs.pop('image_grid_thw', None) is None:
                logger.warning_once("image_grid_thw is not exists.")

            # Run the encoder.
            # `curr_group_outputs` is either of the following:
            # 1. A tensor of shape (num_items, feature_size, hidden_size)
            # in case feature_size is fixed across all multimodal items.
            # 2. A list or tuple (length: num_items) of tensors, each of shape
            # (feature_size, hidden_size) in case the feature size is dynamic
            # depending on the input multimodal items.
            curr_group_outputs = self.runner.get_multimodal_embeddings_fn(
                self.runner.state, image_grid_thw, **batched_mm_inputs)

            sanity_check_mm_encoder_outputs(
                curr_group_outputs,
                expected_num_items=num_items,
            )

            for output in curr_group_outputs:
                encoder_outputs.append(output)

        # Cache the encoder outputs.
        for (mm_hash, pos_info), output in zip(
                mm_hashes_pos,
                encoder_outputs,
        ):
            if req_id not in self.runner.encoder_cache:
                self.runner.encoder_cache[req_id] = {}

            self.runner.encoder_cache[mm_hash] = scatter_mm_placeholders_jax(
                output,
                is_embed=pos_info.is_embed.numpy() if pos_info.is_embed is not None else None,
            )

    def execute_mm_encoder(self, scheduler_output: "VllmSchedulerOutput"):
        if DEV_NEW_MM:
            return self.execute_mm_encoder_dev(scheduler_output)

        import torch
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return

        # Batch the multi-modal inputs.
        mm_kwargs = list[MultiModalKwargsItem]()
        # List of tuple (mm_hash, pos_info)
        mm_hashes_pos = list[tuple[str, PlaceholderRange]]()
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.runner.requests[req_id]
            for mm_input_id in encoder_input_ids:
                mm_feature = req_state.mm_features[mm_input_id]
                mm_hash = mm_feature.identifier
                mm_kwargs.append(mm_feature.data)
                mm_hashes_pos.append((mm_hash, mm_feature.mm_position))

        # Batch mm inputs as much as we can: if a request in the batch has
        # multiple modalities or a different modality than the previous one,
        # we process it separately to preserve item order.
        # FIXME(ywang96): This is a hacky way to deal with multiple modalities
        # in the same batch while still being able to benefit from batching
        # multimodal inputs. The proper solution should be reordering the
        # encoder outputs.
        encoder_outputs = []
        for _, num_items, mm_kwargs_group in group_mm_kwargs_by_modality(
                mm_kwargs, merge_by_field_config=False):
            batched_mm_inputs = mm_kwargs_group
            # Convert torch tensors to numpy arrays that JAX can handle.
            if "pixel_values" in batched_mm_inputs and isinstance(
                    batched_mm_inputs["pixel_values"], list):
                batched_mm_inputs["pixel_values"] = torch.cat(
                    batched_mm_inputs["pixel_values"], dim=0)

            image_grid_thw = ()
            for key, value in batched_mm_inputs.items():
                if isinstance(value, torch.Tensor):
                    if key == 'image_grid_thw':
                        # change it to tuple of tuples to make it hashable for JIT

                        # Shape: (B, N, 3) -> (B*N, 3) -> tuple of tuples
                        grid_thw_tensor = batched_mm_inputs[key]
                        grid_thw_reshaped = grid_thw_tensor.reshape(-1, 3)
                        image_grid_thw = tuple(
                            tuple(row) for row in grid_thw_reshaped.tolist())

                        continue

                    if value.dtype == torch.bfloat16:
                        batched_mm_inputs[key] = value.to(
                            torch.float32).numpy().astype(jnp.bfloat16)
                    else:
                        batched_mm_inputs[key] = value.numpy()
            batched_mm_inputs.pop('image_grid_thw')

            # Run the encoder.
            # `curr_group_outputs` is either of the following:
            # 1. A tensor of shape (num_items, feature_size, hidden_size)
            # in case feature_size is fixed across all multimodal items.
            # 2. A list or tuple (length: num_items) of tensors, each of shape
            # (feature_size, hidden_size) in case the feature size is dynamic
            # depending on the input multimodal items.
            curr_group_outputs = self.runner.get_multimodal_embeddings_fn(
                self.runner.state, image_grid_thw, **batched_mm_inputs)

            sanity_check_mm_encoder_outputs(
                curr_group_outputs,
                expected_num_items=num_items,
            )

            for output in curr_group_outputs:
                encoder_outputs.append(output)

        # Cache the encoder outputs.
        for (mm_hash, pos_info), output in zip(
                mm_hashes_pos,
                encoder_outputs,
        ):
            if req_id not in self.runner.encoder_cache:
                self.runner.encoder_cache[req_id] = {}

            self.runner.encoder_cache[mm_hash] = scatter_mm_placeholders(
                output,
                is_embed=pos_info.is_embed,
            )

    def gather_mm_embeddings_dev(self, scheduler_output: "VllmSchedulerOutput",
                             target_pad_len: int) -> list[jax.Array]:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens

        mm_embeds: list[jax.Array] = []
        is_mm_embed = jnp.zeros((total_num_scheduled_tokens,), dtype=jnp.bool_)

        req_start_idx = 0

        for req_id in self.runner.input_batch.req_ids:
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]
            req_state = self.runner.requests[req_id]
            num_computed_tokens = req_state.num_computed_tokens
            mm_features = req_state.mm_features
            for _, mm_feature in enumerate(mm_features):
                pos_info = mm_feature.mm_position
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length

                # The encoder output is needed if the two ranges overlap:
                # [num_computed_tokens,
                #  num_computed_tokens + num_scheduled_tokens) and
                # [start_pos, start_pos + num_encoder_tokens)
                if start_pos >= num_computed_tokens + num_scheduled_tokens:
                    # The encoder output is not needed in this step.
                    break
                if start_pos + num_encoder_tokens <= num_computed_tokens:
                    # The encoder output is already processed and stored
                    # in the decoder's KV cache.
                    continue

                start_idx = max(num_computed_tokens - start_pos, 0)
                end_idx = min(
                    num_computed_tokens - start_pos + num_scheduled_tokens,
                    num_encoder_tokens)
                assert start_idx < end_idx
                mm_hash = mm_feature.identifier
                encoder_output = self.runner.encoder_cache.get(mm_hash, None)
                assert encoder_output is not None,\
                      f"Encoder cache miss for {mm_hash}."
                encoder_output = self.runner.encoder_cache[mm_hash]

                if (is_embed := pos_info.is_embed) is not None:
                    is_embed = is_embed[start_idx:end_idx]

                req_start_pos = req_start_idx + start_pos - num_computed_tokens
                is_mm_embed = is_mm_embed.at[
                    req_start_pos + start_idx : req_start_pos + end_idx
                ].set(True if is_embed is None else is_embed)

                mm_embeds_item = gather_mm_placeholders(
                    encoder_output[start_idx:end_idx],
                    # is_embed=is_embed,
                    is_embed=is_embed.numpy() if is_embed is not None else None,
                )
                mm_embeds.append(mm_embeds_item)
                req_start_idx += num_scheduled_tokens
        if not mm_embeds:
            return None, None
        flattened_embeds = flatten_embeddings(mm_embeds)
        if flattened_embeds.shape[0] == 0:
            return None, None

        padding = jnp.zeros((target_pad_len - flattened_embeds.shape[0],
                             flattened_embeds.shape[1]),
                            dtype=flattened_embeds.dtype)
        flattened_embeds = jnp.concatenate([flattened_embeds, padding], axis=0)
        is_mm_embed = jnp.pad(is_mm_embed, (0, target_pad_len - is_mm_embed.shape[0]))
        assert flattened_embeds.shape[0] == is_mm_embed.shape[0]

        return flattened_embeds, is_mm_embed

    def gather_mm_embeddings(self, scheduler_output: "VllmSchedulerOutput",
                             target_pad_len: int) -> list[jax.Array]:
        if DEV_NEW_MM:
            return self.gather_mm_embeddings_dev(scheduler_output, target_pad_len)

        mm_embeds: list[jax.Array] = []
        for req_id in self.runner.input_batch.req_ids:
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]
            req_state = self.runner.requests[req_id]
            num_computed_tokens = req_state.num_computed_tokens
            mm_features = req_state.mm_features
            for _, mm_feature in enumerate(mm_features):
                pos_info = mm_feature.mm_position
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length

                # The encoder output is needed if the two ranges overlap:
                # [num_computed_tokens,
                #  num_computed_tokens + num_scheduled_tokens) and
                # [start_pos, start_pos + num_encoder_tokens)
                if start_pos >= num_computed_tokens + num_scheduled_tokens:
                    # The encoder output is not needed in this step.
                    break
                if start_pos + num_encoder_tokens <= num_computed_tokens:
                    # The encoder output is already processed and stored
                    # in the decoder's KV cache.
                    continue

                start_idx = max(num_computed_tokens - start_pos, 0)
                end_idx = min(
                    num_computed_tokens - start_pos + num_scheduled_tokens,
                    num_encoder_tokens)
                assert start_idx < end_idx
                mm_hash = mm_feature.identifier
                encoder_output = self.runner.encoder_cache.get(mm_hash, None)
                assert encoder_output is not None,\
                      f"Encoder cache miss for {mm_hash}."
                encoder_output = self.runner.encoder_cache[mm_hash]

                if (is_embed := pos_info.is_embed) is not None:
                    is_embed = is_embed[start_idx:end_idx]

                mm_embeds_item = gather_mm_placeholders(
                    encoder_output[start_idx:end_idx],
                    is_embed=is_embed,
                )
                mm_embeds.append(mm_embeds_item)
        if not mm_embeds:
            return None
        flattened_embeds = flatten_embeddings(mm_embeds)
        if flattened_embeds.shape[0] == 0:
            return None

        padding = jnp.zeros((target_pad_len - flattened_embeds.shape[0],
                             flattened_embeds.shape[1]),
                            dtype=flattened_embeds.dtype)
        flattened_embeds = jnp.concatenate([flattened_embeds, padding], axis=0)

        return flattened_embeds
