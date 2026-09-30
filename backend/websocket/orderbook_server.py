import asyncio
import json

from confluent_kafka import Consumer
import websockets


KAFKA_SERVER = "localhost:9092"
TOPIC_NAME = "orderbook"
WEBSOCKET_HOST = "localhost"
WEBSOCKET_PORT = 8765


consumer = Consumer({
    "bootstrap.servers": KAFKA_SERVER,
    "group.id": "quantformer-websocket",
    "auto.offset.reset": "latest",
})

consumer.subscribe([TOPIC_NAME])

clients = set()


async def send_orderbook():
    print("Kafka → WebSocket bridge started.")
    print("Listening for order book data...\n")

    try:
        while True:
            message = consumer.poll(0.1)

            if message is None:
                await asyncio.sleep(0.01)
                continue

            if message.error():
                print(f"Kafka error: {message.error()}")
                continue

            data = json.loads(message.value().decode("utf-8"))

            if clients:
                message_text = json.dumps(data)

                await asyncio.gather(
                    *[
                        client.send(message_text)
                        for client in clients
                    ],
                    return_exceptions=True,
                )

    except asyncio.CancelledError:
        pass


async def websocket_handler(websocket):
    clients.add(websocket)

    print("React client connected.")

    try:
        await websocket.wait_closed()

    finally:
        clients.discard(websocket)
        print("React client disconnected.")


async def main():
    async with websockets.serve(
        websocket_handler,
        WEBSOCKET_HOST,
        WEBSOCKET_PORT,
    ):
        print(
            f"WebSocket server running at "
            f"ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}"
        )

        await send_orderbook()


if __name__ == "__main__":
    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        print("\nWebSocket server stopped.")

    finally:
        consumer.close()