
import "./App.css";
import PriceChart from "./components/PriceChart";
import OrderBook from "./components/OrderBook";
import ForecastPanel from "./components/ForecastPanel";
import InterpretabilityPanel from "./components/InterpretabilityPanel";

function App() {
  return (
    <div className="app">
      <header className="dashboard-header">
        <h1>QUANTFORMER</h1>
        <p>Multimodal Order Book &amp; Sentiment Transformer</p>
      </header>

      <main className="dashboard">
        <section className="chart-card">
          <h2>Live Price Chart</h2>
          <PriceChart />
        </section>

        <section className="forecast-section">
          <ForecastPanel />
        </section>

        <section className="interpretability-section">
          <InterpretabilityPanel />
        </section>

        <section className="order-book-card">
          <OrderBook />
        </section>
      </main>
    </div>
  );
}

export default App;
