import { useEffect, useRef } from "react";
import {
  createChart,
  CandlestickSeries,
} from "lightweight-charts";

function PriceChart() {
  const chartContainerRef = useRef(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 450,
      layout: {
        background: {
          type: "solid",
          color: "#111827",
        },
        textColor: "#d1d5db",
      },
      grid: {
        vertLines: {
          color: "#1f2937",
        },
        horzLines: {
          color: "#1f2937",
        },
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: true,
      },
    });

    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderVisible: false,
      wickUpColor: "#22c55e",
      wickDownColor: "#ef4444",
    });

    const data = [
      {
        time: 1760000000,
        open: 100.2,
        high: 100.8,
        low: 99.9,
        close: 100.55,
      },
      {
        time: 1760000060,
        open: 100.55,
        high: 101.1,
        low: 100.3,
        close: 100.95,
      },
      {
        time: 1760000120,
        open: 100.95,
        high: 101.4,
        low: 100.7,
        close: 101.2,
      },
      {
        time: 1760000180,
        open: 101.2,
        high: 101.35,
        low: 100.6,
        close: 100.85,
      },
      {
        time: 1760000240,
        open: 100.85,
        high: 101.5,
        low: 100.7,
        close: 101.35,
      },
      {
        time: 1760000300,
        open: 101.35,
        high: 101.7,
        low: 101.0,
        close: 101.55,
      },
      {
        time: 1760000360,
        open: 101.55,
        high: 101.8,
        low: 101.1,
        close: 101.25,
      },
      {
        time: 1760000420,
        open: 101.25,
        high: 101.6,
        low: 100.9,
        close: 101.45,
      },
    ];

    candlestickSeries.setData(data);

    chart.timeScale().fitContent();

    const handleResize = () => {
      chart.applyOptions({
        width: chartContainerRef.current.clientWidth,
      });
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, []);

  return (
    <div className="chart-container">
      <div ref={chartContainerRef} />
    </div>
  );
}

export default PriceChart;