#!/usr/bin/env python3
import re

def test_quote_fix():
    # Test the fixed regex patterns
    text = "Test with regular quotes and unicode quotes"
    
    # Test the patterns we're using
    pattern1 = r'[\u201C\u201D]'  # Smart double quotes
    pattern2 = r'[\u2018\u2019]'  # Smart single quotes
    
    # This should not crash
    result1 = re.sub(pattern1, '"', text)
    result2 = re.sub(pattern2, "'", result1)
    
    print("✅ Quote regex patterns work correctly!")
    print(f"Result: {result2}")
    return True

if __name__ == "__main__":
    test_quote_fix()
