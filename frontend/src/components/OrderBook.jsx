import { useEffect, useRef, useState } from "react";

function OrderBook() {
  const [orderBook, setOrderBook] = useState(null);
  const [connected, setConnected] = useState(false);
  const [updatesPerSecond, setUpdatesPerSecond] = useState(0);
  const updateCountRef = useRef(0);

  useEffect(() => {
    const socket = new WebSocket("ws://localhost:8765");
    const updateTimer = setInterval(() => {
      setUpdatesPerSecond(updateCountRef.current);
      updateCountRef.current = 0;
    }, 1000);

    socket.onopen = () => {
      console.log("Connected to QuantFormer WebSocket");
      setConnected(true);
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);

      updateCountRef.current += 1;

      setOrderBook(data);
    };

    socket.onerror = (error) => {
      console.error("WebSocket error:", error);
      setConnected(false);
    };

    socket.onclose = () => {
      console.log("WebSocket connection closed");
      setConnected(false);
    };

    return () => {
      clearInterval(updateTimer);
      socket.close();
    };
  }, []);

  if (!orderBook) {
    return (
      <div className="order-book">
        <div className="order-book-header">
          <h2>Order Book</h2>
          <span className="symbol">
            {connected ? "CONNECTING..." : "OFFLINE"}
          </span>
        </div>

        <p>Waiting for live market data...</p>
      </div>
    );
  }

  const asks = orderBook.asks || [];
  const bids = orderBook.bids || [];

  const bestBid = bids.length > 0 ? bids[0].price : 0;
  const bestAsk = asks.length > 0 ? asks[0].price : 0;

  const spread = bestAsk && bestBid
    ? (bestAsk - bestBid).toFixed(2)
    : "0.00";

  return (
    <div className="order-book">
      <div className="order-book-header">
        <h2>Order Book</h2>

        <span className="symbol">
          {orderBook.symbol || "MOCK-STOCK"}
        </span>
      </div>

      <div className="connection-status">
        <span className={connected ? "status-dot online" : "status-dot"}>
          ●
        </span>

        {connected ? "LIVE" : "OFFLINE"}

        {connected && (
          <span className="update-rate">
            {updatesPerSecond} updates/sec
          </span>
        )}
      </div>

      <div className="order-book-columns">
        <span>Price</span>
        <span>Volume</span>
      </div>

      <div className="section-title ask-title">
        ASK
      </div>

      <div className="order-levels">
        {asks.map((level, index) => (
          <div
            className="order-row ask-row"
            key={`ask-${level.price}-${index}`}
          >
            <span>{Number(level.price).toFixed(2)}</span>
            <span>{level.volume}</span>
          </div>
        ))}
      </div>

      <div className="spread-row">
        <span>Spread</span>
        <span>{spread}</span>
      </div>

      <div className="section-title bid-title">
        BID
      </div>

      <div className="order-levels">
        {bids.map((level, index) => (
          <div
            className="order-row bid-row"
            key={`bid-${level.price}-${index}`}
          >
            <span>{Number(level.price).toFixed(2)}</span>
            <span>{level.volume}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default OrderBook;