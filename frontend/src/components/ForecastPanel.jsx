
import { useEffect, useState } from "react";

const API_URL = "http://127.0.0.1:8000/forecast";
const SOCKET_URL = "ws://localhost:8765";

function ForecastPanel() {
  const [forecast, setForecast] = useState(null);
  const [livePrice, setLivePrice] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [socketConnected, setSocketConnected] = useState(false);

  useEffect(() => {
    let active = true;

    async function loadForecast() {
      try {
        const response = await fetch(API_URL);

        if (!response.ok) {
          throw new Error(`Forecast API returned ${response.status}`);
        }

        const data = await response.json();

        if (active) {
          setForecast(data);
          setError("");
        }
      } catch (err) {
        if (active) {
          setError(err.message || "Could not load forecast.");
        }
      } finally {
        if (active) setLoading(false);
      }
    }

    loadForecast();
    const timer = setInterval(loadForecast, 60000);

    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    let socket;
    let reconnectTimer;
    let active = true;

    function connect() {
      if (!active) return;

      socket = new WebSocket(SOCKET_URL);

      socket.onopen = () => {
        if (active) setSocketConnected(true);
      };

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          const price = Number(data.mid_price ?? data.midPrice);

          if (active && Number.isFinite(price)) {
            setLivePrice(price);
          }
        } catch (err) {
          console.error("Could not read live price:", err);
        }
      };

      socket.onerror = () => {
        if (active) setSocketConnected(false);
      };

      socket.onclose = () => {
        if (!active) return;

        setSocketConnected(false);
        reconnectTimer = setTimeout(connect, 3000);
      };
    }

    connect();

    return () => {
      active = false;
      clearTimeout(reconnectTimer);
      if (socket) socket.close();
    };
  }, []);

  if (loading) {
    return <section className="forecast-card">Loading TFT forecast...</section>;
  }

  if (error && !forecast) {
    return (
      <section className="forecast-card">
        <h2>TFT Forecast</h2>
        <p className="forecast-error">{error}</p>
        <p>Check that the forecast API is running on port 8000.</p>
      </section>
    );
  }

  const steps = forecast?.forecast_steps || [];
  const fiveMinute = steps.find((step) => Number(step.minutes_ahead) === 5);

  const historicalPrice = Number(forecast?.last_observed_price);
  const lastPrice =
    livePrice !== null ? livePrice : historicalPrice;

  const predictedPrice = fiveMinute
    ? Number(fiveMinute.predicted_mid_price)
    : null;

  const change =
    Number.isFinite(lastPrice) &&
    predictedPrice !== null &&
    Number.isFinite(predictedPrice)
      ? predictedPrice - lastPrice
      : null;

  return (
    <section className="forecast-card">
      <div className="forecast-header">
        <div>
          <h2>TFT Forecast</h2>
          <p className="forecast-subtitle">
            {forecast?.symbol || "MOCK"} · 5-minute horizon
          </p>
        </div>
        <span className="forecast-badge">EXPERIMENTAL</span>
      </div>

      <div className="forecast-live-status">
        <span className={socketConnected ? "status-dot online" : "status-dot"}>
          ●
        </span>
        {socketConnected ? "LIVE PRICE CONNECTED" : "USING LAST AVAILABLE PRICE"}
      </div>

      <div className="forecast-summary">
        <div>
          <span className="forecast-label">Last observed price</span>
          <strong>
            {Number.isFinite(lastPrice) ? lastPrice.toFixed(2) : "Unavailable"}
          </strong>
        </div>

        <div>
          <span className="forecast-label">Predicted price (+5 min)</span>
          <strong>
            {predictedPrice !== null && Number.isFinite(predictedPrice)
              ? predictedPrice.toFixed(2)
              : "Unavailable"}
          </strong>
        </div>

        <div>
          <span className="forecast-label">Predicted change*</span>
          <strong
            className={
              change === null
                ? ""
                : change >= 0
                  ? "forecast-up"
                  : "forecast-down"
            }
          >
            {change === null
              ? "Unavailable"
              : `${change >= 0 ? "+" : ""}${change.toFixed(2)}`}
          </strong>
        </div>
      </div>

      <h3>Forecast steps</h3>

      <div className="forecast-steps">
        {steps.map((step) => (
          <div className="forecast-step" key={step.minutes_ahead}>
            <span>+{step.minutes_ahead} min</span>
            <strong>
              {Number(step.predicted_mid_price).toFixed(2)}
            </strong>
          </div>
        ))}
      </div>

      <p className="forecast-disclaimer">
        Experimental model output. Forecast steps are generated from the
        latest historical CSV window, not recalculated from the live stream.
        *Predicted change compares the displayed live price with the current
        API forecast. Evaluation has not shown improvement over the
        persistence baseline. Point forecasts only; not trading advice.
      </p>
    </section>
  );
}

export default ForecastPanel;
