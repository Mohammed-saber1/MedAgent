import asyncio
import httpx
import websockets
import json

API_URL = "http://localhost:8080/api/query"
WS_URL = "ws://localhost:8080/ws"

async def test_http():
    print("\n--- Testing HTTP Endpoint ---")
    query = "What is the capital of France?" # Evaluated as simple
    async with httpx.AsyncClient() as client:
        response = await client.post(API_URL, json={"userQuery": query})
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")

async def test_ws():
    print("\n--- Testing WebSocket Endpoint ---")
    query = "What are the latest treatments for Type 2 Diabetes?"
    async with websockets.connect(WS_URL) as websocket:
        await websocket.send(json.dumps({"userQuery": query}))
        
        while True:
            response = await websocket.recv()
            data = json.loads(response)
            
            if data['type'] == 'state_update':
                print(f"Update from: {data['node']}")
            elif data['type'] == 'end':
                print("Workflow Complete")
                break
            elif data['type'] == 'error':
                print(f"Error: {data['message']}")
                break

if __name__ == "__main__":
    asyncio.run(test_http())
    # asyncio.run(test_ws()) # Uncomment to test WS (server must be running)
