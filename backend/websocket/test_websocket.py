import asyncio
import websockets


async def test_connection():
    async with websockets.connect("ws://localhost:8765") as websocket:
        print("WebSocket connected!")

        message = await websocket.recv()

        print("Received order-book message!")
        print(message[:300])


if __name__ == "__main__":
    asyncio.run(test_connection())