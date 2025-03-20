"""Utility functions for JSON handling in OpenManus."""

import json
import json_repair

def repair_json_output(json_str: str) -> str:
    """Repair malformed JSON strings.
    
    Args:
        json_str: The potentially malformed JSON string to repair.
        
    Returns:
        str: A valid JSON string.
    """
    try:
        # First try to parse as-is
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        # If parsing fails, attempt repair
        try:
            return json_repair.repair_json(json_str)
        except Exception as e:
            # If repair fails, return original string
            return json_str