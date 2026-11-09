"""Pinned find_matching_blocks from intel/auto-round@7985c1b0, utils/model.py."""
import re


def find_matching_blocks(model, all_blocks, to_quant_block_names):
    if not to_quant_block_names:
        return all_blocks
    to_quant_block_list = to_quant_block_names
    if isinstance(to_quant_block_names, (list, tuple)):
        return to_quant_block_names
    if isinstance(to_quant_block_names, str):
        to_quant_block_list = [name.strip() for name in to_quant_block_names.split(",")]
    target_blocks = []
    for block_list in all_blocks:
        matched_sublist = []
        for name in to_quant_block_list:
            matches = [block for block in block_list if re.search(name, block)]
            if matches:
                matched_sublist.extend(matches)
        if matched_sublist:
            target_blocks.append(matched_sublist)
    if not target_blocks:
        raise ValueError("No block names matched. Please check the input for to_quant_block_name,"
                         "or set to_quant_block_name to None to automatically match quantizable blocks.")
    return target_blocks
