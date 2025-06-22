#!/usr/bin/env python3
"""
Simple test script to check if a specific Llama 3 model supports logprobs via OpenRouter
"""

import os
import json
from openai import OpenAI
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Initialize OpenAI client for OpenRouter (same as project setup)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

def test_single_model(model_name="meta-llama/llama-3.1-8b-instruct"):
    """Test logprobs support for a single model"""
    print(f"Testing logprobs for: {model_name}")
    print("-" * 50)
    
    try:
        # Simple test prompt
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "Answer yes or no: Is the sky blue?"}
            ],
            max_tokens=10,
            temperature=0,
            logprobs=True,
            top_logprobs=5
        )
        
        print("✅ API call successful!")
        print(f"Response: {response.choices[0].message.content}")
        
        # Check logprobs
        if response.choices[0].logprobs:
            print("✅ Logprobs object exists")
            
            if response.choices[0].logprobs.content:
                content_logprobs = response.choices[0].logprobs.content
                print(f"✅ Content logprobs available ({len(content_logprobs)} tokens)")
                
                # Show details for first token
                first_token = content_logprobs[0]
                print(f"\nFirst token details:")
                print(f"  Token: '{first_token.token}'")
                print(f"  Logprob: {first_token.logprob:.4f}")
                
                # Show alternatives if available
                if hasattr(first_token, 'top_logprobs') and first_token.top_logprobs:
                    print(f"  Alternatives ({len(first_token.top_logprobs)}):")
                    for i, alt in enumerate(first_token.top_logprobs):
                        print(f"    {i+1}. '{alt.token}' (logprob: {alt.logprob:.4f})")
                else:
                    print("  ⚠️ No top_logprobs alternatives")
                
                return True
            else:
                print("❌ No content logprobs returned")
                return False
        else:
            print("❌ No logprobs object returned")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    # Test the default model
    success = test_single_model()
    
    if success:
        print("\n🎉 SUCCESS: Logprobs are supported!")
        print("You can use this model for lie detection with logprobs.")
    else:
        print("\n💡 TIP: Try a different model or check if logprobs are supported")
        print("Alternative models to try:")
        print("- meta-llama/llama-3-8b-instruct")
        print("- meta-llama/llama-3.1-70b-instruct") 
        print("- Or use OpenAI models like gpt-3.5-turbo")