
import { useEffect, useRef, useState } from "react";
import {
  createChart,
  CandlestickSeries,
  LineSeries,
} from "lightweight-charts";

const SOCKET_URL = "ws://localhost:8765";
const API_URL = "http://127.0.0.1:8000/forecast";
const MAX_CANDLES = 300;
const FORECAST_REFRESH_MS = 60000;

function PriceChart() {
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const candleSeriesRef = useRef(null);
  const livePriceLineRef = useRef(null);
  const forecastLineRef = useRef(null);
  const currentCandleRef = useRef(null);
  const candlesRef = useRef([]);
  const lastCandleTimeRef = useRef(null);

  const [connected, setConnected] = useState(false);
  const [lastPrice, setLastPrice] = useState(null);
  const [forecastStatus, setForecastStatus] = useState("Loading forecast...");
  const [forecastValues, setForecastValues] = useState([]);

  // Create the chart and connect to the live order-book stream.
  useEffect(() => {
    const container = chartContainerRef.current;
    if (!container) return;

    const chart = createChart(container, {
      width: container.clientWidth,
      height: 450,
      layout: {
        background: { type: "solid", color: "#111827" },
        textColor: "#d1d5db",
      },
      grid: {
        vertLines: { color: "#1f2937" },
        horzLines: { color: "#1f2937" },
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 8,
        barSpacing: 12,
        minBarSpacing: 5,
      },
      rightPriceScale: {
        borderColor: "#374151",
      },
    });

    const candles = chart.addSeries(CandlestickSeries, {
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderVisible: false,
      wickUpColor: "#22c55e",
      wickDownColor: "#ef4444",
    });

    const livePriceLine = chart.addSeries(LineSeries, {
      color: "#38bdf8",
      lineWidth: 2,
      lastValueVisible: true,
      priceLineVisible: true,
      crosshairMarkerVisible: false,
      title: "Live price",
    });

    const forecastLine = chart.addSeries(LineSeries, {
      color: "#f59e0b",
      lineWidth: 3,
      lineStyle: 2,
      lastValueVisible: true,
      priceLineVisible: false,
      crosshairMarkerVisible: true,
      title: "TFT forecast",
    });

    chartRef.current = chart;
    candleSeriesRef.current = candles;
    livePriceLineRef.current = livePriceLine;
    forecastLineRef.current = forecastLine;

    const socket = new WebSocket(SOCKET_URL);

    socket.onopen = () => setConnected(true);
    socket.onclose = () => setConnected(false);
    socket.onerror = () => setConnected(false);

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const rawPrice = Number(data.mid_price ?? data.midPrice);
        const rawTime = data.timestamp ?? data.time;

        if (!Number.isFinite(rawPrice) || rawTime == null) return;

        const date =
          typeof rawTime === "number"
            ? new Date(rawTime > 1e12 ? rawTime : rawTime * 1000)
            : new Date(rawTime);

        if (Number.isNaN(date.getTime())) return;

        // Use minute-based candle timestamps.
        const time = Math.floor(date.getTime() / 60000) * 60;
        const price = rawPrice;

        setLastPrice(price);

        const current = currentCandleRef.current;

        if (!current || time > current.time) {
          const nextCandle = {
            time,
            open: price,
            high: price,
            low: price,
            close: price,
          };

          currentCandleRef.current = nextCandle;
          lastCandleTimeRef.current = time;
          candlesRef.current.push(nextCandle);

          if (candlesRef.current.length > MAX_CANDLES) {
            candlesRef.current.shift();
            candles.setData(candlesRef.current);
          } else {
            candles.update(nextCandle);
          }

          livePriceLine.update({ time, value: price });
        } else if (time === current.time) {
          current.high = Math.max(current.high, price);
          current.low = Math.min(current.low, price);
          current.close = price;

          candles.update(current);
          livePriceLine.update({ time, value: price });
        }
      } catch (error) {
        console.error("Could not process order-book update:", error);
      }
    };

    const handleResize = () => {
      if (chartRef.current && container) {
        chartRef.current.applyOptions({
          width: container.clientWidth,
        });
      }
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      socket.close();
      chart.remove();

      chartRef.current = null;
      candleSeriesRef.current = null;
      livePriceLineRef.current = null;
      forecastLineRef.current = null;
      currentCandleRef.current = null;
      lastCandleTimeRef.current = null;
      candlesRef.current = [];
    };
  }, []);

  // Load the existing five-step forecast from FastAPI.
  useEffect(() => {
    let active = true;
    let inFlight = false;

    async function loadForecast() {
      if (inFlight) return;
      inFlight = true;

      try {
        const response = await fetch(API_URL);

        if (!response.ok) {
          throw new Error(`Forecast API returned ${response.status}`);
        }

        const data = await response.json();

        const steps = (data.forecast_steps || [])
          .map((step) => ({
            minutesAhead: Number(step.minutes_ahead),
            price: Number(step.predicted_mid_price),
          }))
          .filter(
            (step) =>
              Number.isFinite(step.minutesAhead) &&
              Number.isFinite(step.price) &&
              step.minutesAhead > 0
          )
          .sort((a, b) => a.minutesAhead - b.minutesAhead);

        if (!steps.length) {
          throw new Error("The API returned no valid forecast steps.");
        }

        if (!active) return;

        setForecastValues(steps);
        setForecastStatus(
          "Experimental forecast · historical CSV input"
        );
      } catch (error) {
        if (active) {
          setForecastStatus(`Forecast unavailable: ${error.message}`);
        }
      } finally {
        inFlight = false;
      }
    }

    loadForecast();
    const timer = setInterval(loadForecast, FORECAST_REFRESH_MS);

    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  // Display the forecast after the latest observed candle.
  // The API currently forecasts from historical CSV data, not live ticks.
  useEffect(() => {
    const chart = chartRef.current;
    const series = forecastLineRef.current;
    const lastTime = lastCandleTimeRef.current;

    if (!chart || !series || lastTime == null || !forecastValues.length) {
      return;
    }

    const forecastData = forecastValues.map((step) => ({
      time: lastTime + step.minutesAhead * 60,
      value: step.price,
    }));

    // Lightweight Charts requires strictly increasing, unique times.
    series.setData(forecastData);
    chart.timeScale().fitContent();
  }, [forecastValues, connected, lastPrice]);

  return (
    <div className="chart-container">
      <div className="chart-status">
        <span className={connected ? "status-dot online" : "status-dot"}>
          ●
        </span>
        {connected ? "LIVE PRICE FEED" : "CONNECTING TO PRICE FEED"}

        {lastPrice !== null && (
          <span className="chart-last-price">
            Mid-price: {lastPrice.toFixed(2)}
          </span>
        )}
      </div>

      <div ref={chartContainerRef} />

      <div className="forecast-chart-legend">
        <span>
          <span className="legend-live" /> Live price
        </span>
        <span>
          <span className="legend-forecast" /> TFT forecast
        </span>
        <span className="forecast-chart-status">{forecastStatus}</span>
      </div>
    </div>
  );
}

export default PriceChart;
