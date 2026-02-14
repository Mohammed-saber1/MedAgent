import asyncio
import httpx
import uuid

API_URL = "http://localhost:8000/api/chat"

async def test_http():
    print(f"\n--- Testing HTTP Endpoint: {API_URL} ---")
    
    session_id = str(uuid.uuid4())
    message = "What is the capital of France?" 
    
    payload = {
        "sessionId": session_id,
        "message": message
    }
    
    print(f"Sending payload: {payload}")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(API_URL, json=payload)
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                print(f"Response: {response.json()}")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_http())
