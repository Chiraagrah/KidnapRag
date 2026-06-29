#!/usr/bin/env python3
"""
Check how action parsing works and what causes failures.
"""
import json

def parse_model_output(output_text: str):
    """
    Parse model output in format: "Thought: ...\nAction: ..."
    Returns: (thought, action)
    """
    thought = None
    action = None
    
    if "Thought:" in output_text:
        thought_part = output_text.split("Thought:", 1)[1]
        if "Action:" in thought_part:
            thought = thought_part.split("Action:", 1)[0].strip()
            action = thought_part.split("Action:", 1)[1].strip()
        else:
            thought = thought_part.strip()
            action = None
    elif "Action:" in output_text:
        action = output_text.split("Action:", 1)[1].strip()
        thought = None
    
    return thought or "", action or ""

def analyze_parsing():
    """
    Show examples of how parsing works and what causes failures.
    """
    print("=== Action Parsing Logic ===\n")
    print("The parse_model_output function expects:")
    print("  Format 1: 'Thought: ...\\nAction: ...'")
    print("  Format 2: 'Action: ...' (without Thought)")
    print("\nParsing logic:")
    print("  1. If 'Thought:' exists:")
    print("     - Split by 'Thought:' and take the part after it")
    print("     - If 'Action:' exists in that part:")
    print("       - Split by 'Action:' to get thought and action")
    print("     - Else: only thought, no action")
    print("  2. Else if 'Action:' exists:")
    print("     - Split by 'Action:' and take the part after it")
    print("  3. Return (thought or '', action or '')")
    
    print("\n=== Test Cases ===\n")
    
    test_cases = [
        # Valid cases
        ("Thought: I need to search\nAction: Search[query]", "Valid with Thought and Action"),
        ("Action: Search[query]", "Valid with only Action"),
        ("Thought: I need to search\nAction: Finish[answer]", "Valid Finish action"),
        ("Thought: I need to search\nAction: None", "Valid None action"),
        
        # Failure cases
        ("Thought: I need to search", "Missing Action (will fail)"),
        ("I need to search", "No Thought or Action (will fail)"),
        ("Thought: I need to search\nAction:", "Empty Action (will fail)"),
        ("Thought: I need to search\nAction: ", "Action with only whitespace (will fail)"),
        ("thought: I need to search\naction: Search[query]", "Wrong case (will fail)"),
        ("Thought: I need to search\nAction Search[query]", "Missing colon after Action (will fail)"),
    ]
    
    for output, description in test_cases:
        thought, action = parse_model_output(output)
        status = "✅ PASS" if action else "❌ FAIL"
        print(f"{status} - {description}")
        print(f"  Input: {repr(output[:60])}")
        print(f"  Parsed thought: {repr(thought[:40]) if thought else 'None'}")
        print(f"  Parsed action: {repr(action) if action else 'None'}")
        print()
    
    print("\n=== Real Failure Analysis ===\n")
    print("When action is empty (''), the code does:")
    print("  if not action:")
    print("      print(f'Warning: Failed to generate action at step {i+1} Action: {action}')")
    print("      break")
    print("\nThis means the trajectory will have empty steps: []")

if __name__ == '__main__':
    analyze_parsing()
