import "./App.css";
import PriceChart from "./components/PriceChart";
import OrderBook from "./components/OrderBook";

function App() {
  return (
    <div className="app">
      <header className="dashboard-header">
        <h1>QUANTFORMER</h1>
        <p>Multimodal Order Book & Sentiment Transformer</p>
      </header>

      <main className="dashboard">
        <section className="chart-card">
          <h2>Live Price Chart</h2>
          <PriceChart />
        </section>

        <section className="order-book-card">
          <OrderBook />
        </section>
      </main>
    </div>
  );
}

export default App;