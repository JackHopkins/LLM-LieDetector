#!/usr/bin/env python3
"""
Test script to evaluate logprobs support for Llama 3 models via OpenRouter
using the OpenAI client (same setup as in the main project).
"""

import os
import json
from openai import OpenAI
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Initialize OpenAI client for OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Llama 3 models to test on OpenRouter
LLAMA3_MODELS = [
    "meta-llama/llama-3-8b-instruct",
    "meta-llama/llama-3-70b-instruct", 
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-405b-instruct"
]

def test_model_basic(model_name):
    """Test basic model functionality without logprobs"""
    print(f"\n=== Testing basic functionality for {model_name} ===")
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "What is 2+2?"}],
            max_tokens=50,
            temperature=0
        )
        
        print(f"✅ Basic call successful")
        print(f"Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ Basic call failed: {e}")
        return False

def test_model_logprobs(model_name):
    """Test logprobs functionality for a model"""
    print(f"\n=== Testing logprobs for {model_name} ===")
    
    # Test 1: Basic logprobs
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Answer with just 'yes' or 'no': Is the sky blue?"}],
            max_tokens=10,
            temperature=0,
            logprobs=True
        )
        
        print(f"✅ Basic logprobs=True successful")
        
        # Check if logprobs data exists
        if response.choices[0].logprobs:
            print(f"✅ Logprobs data returned")
            content_logprobs = response.choices[0].logprobs.content
            if content_logprobs:
                print(f"✅ Content logprobs available ({len(content_logprobs)} tokens)")
                # Show first token details
                first_token = content_logprobs[0]
                print(f"   First token: '{first_token.token}' (logprob: {first_token.logprob:.4f})")
            else:
                print(f"⚠️  No content logprobs returned")
        else:
            print(f"⚠️  No logprobs object returned")
            
    except Exception as e:
        print(f"❌ Basic logprobs failed: {e}")
        return False
    
    # Test 2: Top logprobs
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Answer with just 'yes' or 'no': Is the sky blue?"}],
            max_tokens=10,
            temperature=0,
            logprobs=True,
            top_logprobs=5
        )
        
        print(f"✅ top_logprobs=5 successful")
        
        # Check top logprobs data
        if response.choices[0].logprobs and response.choices[0].logprobs.content:
            content_logprobs = response.choices[0].logprobs.content
            first_token = content_logprobs[0]
            if hasattr(first_token, 'top_logprobs') and first_token.top_logprobs:
                print(f"✅ Top logprobs available ({len(first_token.top_logprobs)} alternatives)")
                # Show top alternatives for first token
                for i, alt in enumerate(first_token.top_logprobs[:3]):
                    print(f"   Alt {i+1}: '{alt.token}' (logprob: {alt.logprob:.4f})")
            else:
                print(f"⚠️  No top_logprobs data returned")
        
    except Exception as e:
        print(f"❌ top_logprobs failed: {e}")
        return False
    
    return True

def test_yes_no_logprobs(model_name):
    """Test specific yes/no logprobs (relevant for lie detection)"""
    print(f"\n=== Testing yes/no token logprobs for {model_name} ===")
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "Answer only 'yes' or 'no': Are you sure about your previous answer?"}
            ],
            max_tokens=5,
            temperature=0,
            logprobs=True,
            top_logprobs=10
        )
        
        if response.choices[0].logprobs and response.choices[0].logprobs.content:
            content_logprobs = response.choices[0].logprobs.content
            
            # Look for yes/no tokens in the response
            yes_no_found = False
            for token_data in content_logprobs:
                token = token_data.token.lower().strip()
                if 'yes' in token or 'no' in token:
                    yes_no_found = True
                    print(f"✅ Found yes/no token: '{token_data.token}' (logprob: {token_data.logprob:.4f})")
                    
                    # Check alternatives
                    if hasattr(token_data, 'top_logprobs') and token_data.top_logprobs:
                        print(f"   Alternatives:")
                        for alt in token_data.top_logprobs:
                            print(f"     '{alt.token}' (logprob: {alt.logprob:.4f})")
                    break
            
            if not yes_no_found:
                print(f"⚠️  No clear yes/no tokens found in response")
                print(f"   Response: {response.choices[0].message.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ Yes/no logprobs test failed: {e}")
        return False

def main():
    """Run comprehensive logprobs tests"""
    print("🚀 Testing Llama 3 models logprobs support via OpenRouter")
    print("=" * 60)
    
    results = {}
    
    for model in LLAMA3_MODELS:
        print(f"\n{'='*20} {model} {'='*20}")
        
        # Test basic functionality first
        basic_works = test_model_basic(model)
        if not basic_works:
            results[model] = {"basic": False, "logprobs": False, "yes_no": False}
            continue
        
        # Test logprobs functionality
        logprobs_works = test_model_logprobs(model)
        
        # Test yes/no specific logprobs
        yes_no_works = False
        if logprobs_works:
            yes_no_works = test_yes_no_logprobs(model)
        
        results[model] = {
            "basic": basic_works,
            "logprobs": logprobs_works, 
            "yes_no": yes_no_works
        }
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY RESULTS")
    print(f"{'='*60}")
    
    for model, result in results.items():
        status = "✅" if result["logprobs"] else "❌"
        yes_no_status = "✅" if result["yes_no"] else "❌"
        print(f"{status} {model}")
        print(f"   Basic: {'✅' if result['basic'] else '❌'}")
        print(f"   Logprobs: {'✅' if result['logprobs'] else '❌'}")  
        print(f"   Yes/No Detection: {yes_no_status}")
        print()
    
    # Recommendations
    working_models = [m for m, r in results.items() if r["logprobs"]]
    if working_models:
        print("🎉 RECOMMENDED MODELS FOR LIE DETECTION:")
        for model in working_models:
            print(f"   - {model}")
    else:
        print("⚠️  No models found with working logprobs support")
        print("   You may need to use the completion-based approach without logprobs")

if __name__ == "__main__":
    main()