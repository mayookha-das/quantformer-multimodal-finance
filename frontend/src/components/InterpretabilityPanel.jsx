import { useEffect, useState } from "react";

const API_URL = "http://127.0.0.1:8000/interpretability";

function FeatureBars({ title, features = [] }) {
  const validFeatures = features.filter(
    (item) =>
      item &&
      typeof item.feature === "string" &&
      Number.isFinite(Number(item.importance))
  );

  const maxValue = Math.max(
    0.000001,
    ...validFeatures.map((item) => Number(item.importance))
  );

  return (
    <div className="interpretability-group">
      <h3>{title}</h3>

      {validFeatures.length === 0 ? (
        <p className="interpretability-muted">
          No feature importance data available.
        </p>
      ) : (
        validFeatures.map((item) => {
          const value = Math.max(0, Number(item.importance));
          const percentage = value * 100;
          const width = (value / maxValue) * 100;

          return (
            <div className="importance-row" key={item.feature}>
              <div className="importance-label">
                <span>{item.feature.replaceAll("_", " ")}</span>
                <strong>{percentage.toFixed(2)}%</strong>
              </div>

              <div
                className="importance-track"
                role="progressbar"
                aria-label={`${item.feature} importance`}
                aria-valuenow={percentage}
                aria-valuemin={0}
                aria-valuemax={100}
              >
                <div
                  className="importance-fill"
                  style={{ width: `${width}%` }}
                />
              </div>
            </div>
          );
        })
      )}
    </div>
  );
}

function AttentionChart({ weights = [] }) {
  const validWeights = weights
    .map(Number)
    .filter(Number.isFinite);

  if (validWeights.length === 0) {
    return (
      <p className="interpretability-muted">
        No attention weights available.
      </p>
    );
  }

  const maxWeight = Math.max(0.000001, ...validWeights);

  return (
    <div className="attention-chart">
      {validWeights.map((weight, index) => (
        <div className="attention-column" key={index}>
          <div className="attention-track">
            <div
              className="attention-fill"
              style={{
                height: `${(weight / maxWeight) * 100}%`,
              }}
              title={`Weight ${weight.toFixed(4)}`}
            />
          </div>
          <span>{index + 1}</span>
        </div>
      ))}
    </div>
  );
}

function InterpretabilityPanel() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function loadInterpretability() {
      try {
        const response = await fetch(API_URL);

        if (!response.ok) {
          throw new Error(
            `Interpretability API returned ${response.status}`
          );
        }

        const result = await response.json();

        if (active) {
          setData(result);
          setError("");
        }
      } catch (err) {
        if (active) {
          setError(err.message || "Could not load interpretability data.");
        }
      } finally {
        if (active) setLoading(false);
      }
    }

    loadInterpretability();

    const timer = setInterval(loadInterpretability, 60000);

    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  return (
    <section className="interpretability-card">
      <div className="interpretability-header">
        <div>
          <h2>Model Interpretability</h2>
          <p className="interpretability-subtitle">
            TFT feature importance · {data?.symbol || "MOCK"}
          </p>
        </div>

        <span className="interpretability-badge">EXPERIMENTAL</span>
      </div>

      {loading && <p>Loading model interpretation...</p>}

      {!loading && error && (
        <div className="interpretability-error">
          <p>{error}</p>
          <p>Check that the Forecast API is running on port 8000.</p>
        </div>
      )}

      {!loading && !error && data && (
        <>
          <FeatureBars
            title="Historical input importance"
            features={data.encoder_feature_importance}
          />

          <FeatureBars
            title="Decoder feature importance"
            features={data.decoder_feature_importance}
          />

          <div className="interpretability-group">
            <h3>Historical attention weights</h3>
            <p className="interpretability-muted">
              Relative attention values across the returned historical
              window. Positions are ordered as returned by the API.
            </p>
            <AttentionChart weights={data.attention_weights} />
          </div>

          <p className="interpretability-disclaimer">
            {data.note ||
              "Experimental model interpretation. Importance and attention do not establish causation or trading value."}
          </p>
        </>
      )}
    </section>
  );
}

export default InterpretabilityPanel;