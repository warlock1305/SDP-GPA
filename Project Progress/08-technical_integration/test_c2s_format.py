"""
Test script to understand .c2s file format
"""

import os

# Test with a small sample from the .c2s file
c2s_file = "ExtractedPaths/ibecir/oop-1002/java/java/data/path_contexts.c2s"

if os.path.exists(c2s_file):
    print("=== Testing .c2s file format ===")
    
    with open(c2s_file, 'r', encoding='utf-8') as f:
        # Read first few lines to understand format
        for i, line in enumerate(f):
            if i >= 5:  # Only read first 5 lines
                break
            print(f"Line {i+1}: {repr(line.strip())}")
            
            # Try to parse the line
            if line.strip():
                parts = line.strip().split(' ')
                print(f"  Parts: {len(parts)}")
                if len(parts) >= 2:
                    method_name = parts[0]
                    path_contexts = parts[1:]
                    print(f"  Method: {method_name}")
                    print(f"  Contexts: {len(path_contexts)}")
                    
                    # Check first few contexts
                    for j, context in enumerate(path_contexts[:3]):
                        print(f"    Context {j+1}: {repr(context)}")
                        if context.startswith('|') and context.endswith('|'):
                            content = context[1:-1]
                            if ',' in content:
                                context_parts = content.split(',')
                                print(f"      Parsed: {context_parts}")
else:
    print(f"File not found: {c2s_file}")
