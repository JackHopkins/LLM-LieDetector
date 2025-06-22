#!/usr/bin/env python3
"""
Debug script to understand the exact logprobs structure returned by Llama models
"""

import os
import dotenv
from openai import OpenAI

# Load environment variables
dotenv.load_dotenv()

# Initialize client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

def debug_logprobs_structure(model_name="meta-llama/llama-3.1-8b-instruct"):
    """Debug the exact structure of logprobs returned"""
    print(f"🔍 DEBUGGING LOGPROBS STRUCTURE FOR {model_name}")
    print("="*60)
    
    # Test 1: Only logprobs=True
    print("\n1. Testing logprobs=True only:")
    try:
        response1 = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Answer: What is 2+2?"}],
            max_tokens=10,
            temperature=0,
            logprobs=True
        )
        
        choice = response1.choices[0]
        print(f"   Response: {choice.message.content}")
        print(f"   Has logprobs: {choice.logprobs is not None}")
        print(f"   Logprobs object: {choice.logprobs}")
        if choice.logprobs:
            print(f"   Has content: {choice.logprobs.content is not None}")
            if choice.logprobs.content:
                print(f"   Content length: {len(choice.logprobs.content)}")
                if choice.logprobs.content:
                    first_token = choice.logprobs.content[0]
                    print(f"   First token: {first_token.token}")
                    print(f"   First logprob: {first_token.logprob}")
                    print(f"   Has top_logprobs: {hasattr(first_token, 'top_logprobs') and first_token.top_logprobs is not None}")
            else:
                print("   ⚠️  Content is None/empty")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: logprobs=True + top_logprobs=5  
    print("\n2. Testing logprobs=True + top_logprobs=5:")
    try:
        response2 = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Answer with just 'yes' or 'no': Is the sky blue?"}],
            max_tokens=10,
            temperature=0,
            logprobs=True,
            top_logprobs=5
        )
        
        choice = response2.choices[0]
        print(f"   Response: {choice.message.content}")
        print(f"   Has logprobs: {choice.logprobs is not None}")
        print(f"   Logprobs object: {choice.logprobs}")
        if choice.logprobs:
            print(f"   Has content: {choice.logprobs.content is not None}")
            if choice.logprobs.content:
                print(f"   Content length: {len(choice.logprobs.content)}")
                first_token = choice.logprobs.content[0]
                print(f"   First token: {first_token.token}")
                print(f"   First logprob: {first_token.logprob}")
                print(f"   Has top_logprobs: {hasattr(first_token, 'top_logprobs') and first_token.top_logprobs is not None}")
                if first_token.top_logprobs:
                    print(f"   Top logprobs count: {len(first_token.top_logprobs)}")
                    print(f"   Top alternatives: {[(alt.token, alt.logprob) for alt in first_token.top_logprobs[:3]]}")
            else:
                print("   ⚠️  Content is None/empty")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    debug_logprobs_structure()