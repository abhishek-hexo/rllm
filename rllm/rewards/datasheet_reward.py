import re
import string
from collections import Counter
from typing import Any

import json
from rllm.rewards.reward_types import RewardConfig, RewardInput, RewardOutput
import requests
from openai import OpenAI



class RewardDatasheetFn:

    SYSTEM_PROMPT = """
    f"You are a helpful assistant that understands and translates text to JSON format according to the following schema. {json_schema}. If you don't have answer for any field, leave it as empty string."
    """

    JSON_SCHEMAS = {
        "block_diagram": """
        {
        "type": "object",
        "properties": {
            "pins": {
            "type": "array",
            "items": {
                "type": "string"
            }
            },
            "blocks": {
            "type": "array",
            "items": {
                "type": "string"
            }
            }
        },
        "required": ["pins", "blocks"]
        }
        """,
        "pin_map": """
        {
        "type": "object",
        "properties": {
            "pins": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                "pin": {
                    "type": "string"
                },
                "description": {
                    "type": "string"
                },
                "type": {
                    "type": "string",
                    "enum": ["input", "output"]
                },
                "number": {
                    "type": "string"
                }
                },
                "required": ["pin", "description", "type", "number"]
            }
            }
        },
        "required": ["pins"]
}
        """,
        "register_map": """
        {
        "type": "object",
        "properties": {
            "address": { "type": "string" },
            "type": { "type": "string" },
            "description": { "type": "string" },
            "size": { "type": "integer" },
            "default_value": { "type": "string" },
            "bit_fields": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                "name": { "type": "string" },
                "description": { "type": "string" },
                "size": { "type": "integer" },
                "default_value": { "type": "string" },
                "start_bit": { "type": "integer" },
                "end_bit": { "type": "integer" },
                "type": { "type": "string" }
                },
                "required": [
                "name",
                "description",
                "size",
                "default_value",
                "start_bit",
                "end_bit",
                "type"
                ]
            }
            }
        },
        "required": [
            "address",
            "type",
            "description",
            "size",
            "default_value",
            "bit_fields"
        ]
        }
        """
    }

    client = OpenAI(base_url="http://localhost:8006/v1", api_key="abc")


    def __init__(self, config: RewardConfig):
        self.config = config
        # Formatting reward when decoded JSON matches the expected schema.
        self.format_reward: float = 0.5

    def _strip_code_fences(self, s: str) -> str:
        s = (s or "").strip()
        s = re.sub(r"^`+\w*\s*\n?", "", s)
        s = re.sub(r"\n?`+\s*$", "", s)
        return s.strip()

    def _maybe_json_loads(self, payload: Any, *, strip_fences: bool = False) -> Any:
        """
        Accepts either:
        - a JSON string (optionally fenced), or
        - an already-decoded python object (dict/list/etc.)
        """
        if isinstance(payload, str):
            s = self._strip_code_fences(payload) if strip_fences else payload
            return json.loads(s)
        return payload

    def _is_block_diagram_format(self, obj: Any) -> bool:
        if not isinstance(obj, dict):
            return False
        pins = obj.get("pins")
        blocks = obj.get("blocks")
        if not isinstance(pins, list) or not isinstance(blocks, list):
            return False
        if not all(isinstance(x, str) for x in pins):
            return False
        if not all(isinstance(x, str) for x in blocks):
            return False
        return True

    def _is_pin_map_format(self, obj: Any) -> bool:
        # Expected: list[{"pin": str, "type": str, "number": str, ...}], sometimes wrapped as {"pins": [...]}
        if isinstance(obj, dict) and "pins" in obj:
            obj = obj["pins"]
        if not isinstance(obj, list):
            return False
        for row in obj:
            if not isinstance(row, dict):
                return False
            if not isinstance(row.get("pin"), str) or not row.get("pin", "").strip():
                return False
            if not isinstance(row.get("type"), str) or not row.get("type", "").strip():
                return False
            # number might be str or int in practice; accept both
            num = row.get("number") if "number" in row else row.get("pin_number")
            if not isinstance(num, (str, int)) or str(num).strip() == "":
                return False
        return True

    def _is_register_map_format(self, obj: Any) -> bool:
        # Expected: register dict or list of register dicts.
        # Dataset ground truth may wrap it as {"register_map": ...}
        if isinstance(obj, dict) and "register_map" in obj:
            obj = obj["register_map"]
        if isinstance(obj, dict):
            regs = [obj]
        elif isinstance(obj, list):
            regs = obj
        else:
            return False
        for reg in regs:
            if not isinstance(reg, dict):
                return False
            if not isinstance(reg.get("address"), str) or not reg.get("address", "").strip():
                return False
            if not isinstance(reg.get("size"), (int, str)) or str(reg.get("size")).strip() == "":
                return False
            bfs = reg.get("bit_fields")
            if not isinstance(bfs, list):
                return False
            for bf in bfs:
                if not isinstance(bf, dict):
                    return False
                if not isinstance(bf.get("name"), str) or not bf.get("name", "").strip():
                    return False
                for k in ("size", "start_bit", "end_bit"):
                    if k not in bf or not isinstance(bf.get(k), (int, str)) or str(bf.get(k)).strip() == "":
                        return False
        return True

    def evaluate_block_diagram_answer(self, model_response: str, ground_truth: Any) -> RewardOutput:
        metadata: dict[str, Any] = {}
        try:
            model_obj = self._maybe_json_loads(model_response, strip_fences=True)
            gt_obj = self._maybe_json_loads(ground_truth, strip_fences=True)
        except Exception as e:
            return RewardOutput(reward=0.0, is_correct=False, metadata={"error": f"Invalid JSON: {e}"})


        predicted_pins = model_obj.get("pins") if isinstance(model_obj, dict) else None
        predicted_blocks = model_obj.get("blocks") if isinstance(model_obj, dict) else None
        ground_truth_pins = gt_obj.get("pins") if isinstance(gt_obj, dict) else None
        ground_truth_blocks = gt_obj.get("blocks") if isinstance(gt_obj, dict) else None

        if not (isinstance(predicted_pins, list) and isinstance(predicted_blocks, list) and isinstance(ground_truth_pins, list) and isinstance(ground_truth_blocks, list)):
            return RewardOutput(
                reward=0.0,
                is_correct=False,
                metadata={"error": "Decoded JSON does not contain expected keys/types for block diagram."},
            )

        predicted_pins_set = set(str(p).strip().lower() for p in predicted_pins)
        predicted_blocks_set = set(str(b).strip().lower() for b in predicted_blocks)
        ground_truth_pins_set = set(str(p).strip().lower() for p in ground_truth_pins)
        ground_truth_blocks_set = set(str(b).strip().lower() for b in ground_truth_blocks)

        # Compute Jaccard similarity between predicted and ground truth pins and blocks
        jaccard_similarity_pins = len(predicted_pins_set & ground_truth_pins_set) / len(predicted_pins_set | ground_truth_pins_set) if (predicted_pins_set | ground_truth_pins_set) else 1.0
        jaccard_similarity_blocks = len(predicted_blocks_set & ground_truth_blocks_set) / len(predicted_blocks_set | ground_truth_blocks_set) if (predicted_blocks_set | ground_truth_blocks_set) else 1.0
        metadata["jaccard_similarity_pins"] = jaccard_similarity_pins
        metadata["jaccard_similarity_blocks"] = jaccard_similarity_blocks
        metadata["evaluation_method"] = "jaccard_similarity"
        metadata["predicted_pins"] = predicted_pins
        metadata["predicted_blocks"] = predicted_blocks
        metadata["ground_truth_pins"] = ground_truth_pins
        metadata["ground_truth_blocks"] = ground_truth_blocks
        reward_score = jaccard_similarity_pins + jaccard_similarity_blocks 
        is_correct = jaccard_similarity_blocks >= 0.9 and jaccard_similarity_pins >= 0.9

        return RewardOutput(reward=reward_score, is_correct=is_correct, metadata=metadata)
    
    def evaluate_pin_map_answer(self, model_response: str, ground_truth: Any) -> RewardOutput:
        metadata: dict[str, Any] = {}

        def _norm_pin(x: Any) -> str:
            return str(x).strip().lower() if x is not None else ""

        try:
            predicted = self._maybe_json_loads(model_response, strip_fences=True)
            gt = self._maybe_json_loads(ground_truth, strip_fences=True)
        except Exception as e:
            return RewardOutput(reward=0.0, is_correct=False, metadata={"error": f"Invalid JSON: {e}"})

        # Accept either a raw list [...] or {"pins": [...]}
        if isinstance(predicted, dict) and "pins" in predicted:
            predicted = predicted["pins"]
        if isinstance(gt, dict) and "pins" in gt:
            gt = gt["pins"]

        if not isinstance(predicted, list) or not isinstance(gt, list):
            return RewardOutput(
                reward=0.0,
                is_correct=False,
                metadata={
                    "error": "Expected predicted/ground_truth to be a list (or an object with key 'pins' containing a list).",
                    "predicted_type": type(predicted).__name__,
                    "ground_truth_type": type(gt).__name__,
                },
            )

        def _get_pin_names(rows: list[Any]) -> set[str]:
            pins = set()
            for r in rows:
                if isinstance(r, dict):
                    pin = _norm_pin(r.get("pin") or r.get("name"))
                    if pin:
                        pins.add(pin)
                else:
                    # Not a dict row; skip (could note as metadata if desired)
                    continue
            return pins

        pred_pins = _get_pin_names(predicted)
        gt_pins = _get_pin_names(gt)
        inter = pred_pins & gt_pins
        union = pred_pins | gt_pins

        jaccard_similarity_pins = (len(inter) / len(union)) if union else 1.0

        metadata.update(
            {
                "evaluation_method": "pin_map_jaccard_only",
                "jaccard_similarity_pins": jaccard_similarity_pins,
                "predicted_pin_count": len(pred_pins),
                "ground_truth_pin_count": len(gt_pins),
                "intersection_count": len(inter),
                "union_count": len(union),
                "missing_in_predicted": sorted(list(gt_pins - pred_pins))[:50],
                "extra_in_predicted": sorted(list(pred_pins - gt_pins))[:50],
            }
        )

        score = jaccard_similarity_pins

        is_correct = (jaccard_similarity_pins >= 0.9)

        return RewardOutput(reward=score, is_correct=is_correct, metadata=metadata)

    def evaluate_register_map_answer(self, model_response: str, ground_truth: Any) -> RewardOutput:
        metadata: dict[str, Any] = {}

        def _norm_addr(x: Any) -> str:
            # Keep as string but normalize case/whitespace
            return str(x).strip().lower().replace(" ", "") if x is not None else ""

        def _norm_name(x: Any) -> str:
            return str(x).strip().lower() if x is not None else ""

        try:
            predicted = self._maybe_json_loads(model_response, strip_fences=True)
            gt = self._maybe_json_loads(ground_truth, strip_fences=True)
        except Exception as e:
            return RewardOutput(reward=0.0, is_correct=False, metadata={"error": f"Invalid JSON: {e}"})

        def _unwrap_registers(x: Any) -> list[dict[str, Any]]:
            """
            Accepts formats like:
            - { "address": "...", "bit_fields": [...] }  (single register dict)
            - [{...}, {...}]                             (list of register dicts)
            - { "registers": [...] } or { "register_map": [...] } (wrapped list)
            """
            if isinstance(x, dict):
                if "registers" in x:
                    x = x["registers"]
                elif "register_map" in x:
                    x = x["register_map"]
            if isinstance(x, dict):
                return [x]
            if isinstance(x, list):
                return [r for r in x if isinstance(r, dict)]
            return []

        def _bitfield_name_set(reg: dict[str, Any]) -> set[str]:
            bfs = reg.get("bit_fields")
            if not isinstance(bfs, list):
                return set()
            return {
                _norm_name(bf.get("name"))
                for bf in bfs
                if isinstance(bf, dict) and _norm_name(bf.get("name"))
            }

        predicted_regs = _unwrap_registers(predicted)
        gt_regs = _unwrap_registers(gt)

        # Build address -> union(set(bitfield names)) for robustness if duplicates exist.
        pred_by_addr: dict[str, set[str]] = {}
        for r in predicted_regs:
            addr = _norm_addr(r.get("address"))
            if not addr:
                continue
            pred_by_addr.setdefault(addr, set()).update(_bitfield_name_set(r))

        gt_by_addr: dict[str, set[str]] = {}
        for r in gt_regs:
            addr = _norm_addr(r.get("address"))
            if not addr:
                continue
            gt_by_addr.setdefault(addr, set()).update(_bitfield_name_set(r))

        pred_addr_keys = set(pred_by_addr.keys())
        gt_addr_keys = set(gt_by_addr.keys())
        matching_addresses = pred_addr_keys & gt_addr_keys

        # Per request: string match address, then Jaccard over bit_fields[*].name only.
        best_addr = None
        best_jaccard = 0.0
        best_intersection: set[str] = set()
        best_union: set[str] = set()

        for addr in matching_addresses:
            p_names = pred_by_addr.get(addr, set())
            g_names = gt_by_addr.get(addr, set())
            union = p_names | g_names
            inter = p_names & g_names
            jaccard = (len(inter) / len(union)) if union else 1.0
            if jaccard > best_jaccard:
                best_jaccard = jaccard
                best_addr = addr
                best_intersection = inter
                best_union = union

        has_matching_address = best_addr is not None
        address_match_reward = 0.5 if has_matching_address else 0.0
        bitfield_jaccard = best_jaccard if has_matching_address else 0.0
        reward = address_match_reward + bitfield_jaccard
        is_correct = bool(has_matching_address and bitfield_jaccard >= 0.9)

        metadata.update(
            {
                "evaluation_method": "register_map_address_match_then_bitfield_name_jaccard",
                "has_matching_address": has_matching_address,
                "address_match_reward": address_match_reward,
                "matched_address": best_addr,
                "bitfield_jaccard": bitfield_jaccard,
                "reward_breakdown": {"address": address_match_reward, "bitfield_jaccard": bitfield_jaccard},
                "predicted_address_count": len(pred_addr_keys),
                "ground_truth_address_count": len(gt_addr_keys),
                "matching_addresses": sorted(matching_addresses),
                "missing_addresses_in_predicted": sorted(list(gt_addr_keys - pred_addr_keys))[:50],
                "extra_addresses_in_predicted": sorted(list(pred_addr_keys - gt_addr_keys))[:50],
                "predicted_bitfield_name_count": len(pred_by_addr.get(best_addr, set())) if best_addr else 0,
                "ground_truth_bitfield_name_count": len(gt_by_addr.get(best_addr, set())) if best_addr else 0,
                "intersection_count": len(best_intersection) if has_matching_address else 0,
                "union_count": len(best_union) if has_matching_address else 0,
                "missing_bitfields_in_predicted": sorted(list(gt_by_addr.get(best_addr, set()) - pred_by_addr.get(best_addr, set())))[:50]
                if best_addr
                else [],
                "extra_bitfields_in_predicted": sorted(list(pred_by_addr.get(best_addr, set()) - gt_by_addr.get(best_addr, set())))[:50]
                if best_addr
                else [],
            }
        )

        return RewardOutput(reward=reward, is_correct=is_correct, metadata=metadata)
    

    def convert_answer_to_json(self, model_response: str, source: str) -> Any:
        json_schema = self.JSON_SCHEMAS[source]
        system_prompt_input = self.SYSTEM_PROMPT.format(json_schema=json_schema)
        # query the model hosted in localhost:8006 
        response = self.client.chat.completions.create(
            model="osmosis",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt_input
                },
                {
                    "role": "user", 
                    "content": model_response,
                },
            ],
            temperature=0,
            max_tokens=8192,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "answer_extraction", "schema": json.loads(json_schema)},
            },
        )
        json_response = response.choices[0].message.content
        json_response = re.sub(r"^`+\w*\s*\n?", "", json_response)
        json_response = re.sub(r"\n?`+\s*$", "", json_response)
        return json.loads(json_response)

    def __call__(self, input: RewardInput) -> RewardOutput:
        model_response = input.action
        ground_truth = input.task_info.get("ground_truth") or input.task_info.get("answer")
        data_source = input.task_info.get("data_source") or input.task_info.get("type")
        try:
            json_model_response = self.convert_answer_to_json(model_response, data_source)
        except Exception as e:
            return RewardOutput(reward=0.0, is_correct=False, metadata={"error": str(e)})
        if ground_truth is None:
            return RewardOutput(reward=0.0, is_correct=False, metadata={"error": "No ground truth provided"})
        if data_source == "block_diagram":
            return self.evaluate_block_diagram_answer(json_model_response, ground_truth)
        if data_source == "pin_map":
            return self.evaluate_pin_map_answer(json_model_response, ground_truth)
        if data_source == "register_map":
            return self.evaluate_register_map_answer(json_model_response, ground_truth)

        return RewardOutput(reward=0.0, is_correct=False, metadata={"error": f"Unknown datasheet data_source: {data_source}"})
