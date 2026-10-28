# QuantFormer - Project Status

## Project
**QuantFormer: Multimodal Order Book & Sentiment Transformer**

Domain: Quantitative Finance & Time-Series Forecasting

## Objective

QuantFormer is designed to combine high-frequency Level 2 order-book data with financial-news sentiment for short-term market forecasting.

The development prototype uses mock market and news streams to build and validate the complete architecture.

---

## Current Progress

### Week 1 - Data Engineering

- [x] Kafka 4.3.1 configured using KRaft
- [x] Kafka broker running on localhost:9092
- [x] `orderbook` Kafka topic created
- [x] `financial-news` Kafka topic created
- [x] Mock L2 order-book producer implemented
- [x] Order-book Kafka consumer implemented
- [x] Order-book feature extraction implemented

### Current Order-Book Features

The streaming pipeline currently calculates:

- Mid price
- Best bid
- Best ask
- Bid-ask spread
- Total bid volume
- Total ask volume
- Order-book imbalance

### Streaming Architecture

```text
Mock L2 Order Book
        |
        v
Order Book Producer
        |
        v
Kafka - orderbook topic
        |
        v
Order Book Consumer
        |
        v
Feature Extraction
        |
        v
Quantitative Order-Book Features

## Week 2 — Financial News Sentiment

- [x] Implemented FinBERT-based financial headline sentiment classification.
- [x] Generated sentiment labels, signed sentiment scores, and confidence values.
- [x] Implemented text-embedding generation in the news-processing pipeline.
- [x] Tested the consumer with the Kafka financial-news stream.

## Week 3 — Forecasting

- [x] Prepared time-series data for TFT experiments.
- [x] Implemented a PyTorch Temporal Fusion Transformer training workflow.
- [x] Added an evaluation workflow to compare forecasts with a persistence baseline.
- [x] Reviewed preliminary evaluation results and identified alignment and accuracy limitations.

## Week 4 — Dashboard and Final Integration

- [x] Developed the React dashboard for order-book and price visualization.
- [x] Integrated dashboard updates with backend services.
- [x] Added forecast output and interpretability-related UI components.
- [x] Prepared final project documentation and a demo video plan.

## Final Status and Limitations

QuantFormer is an experimental prototype using simulated market and news data. Kafka streaming, FinBERT sentiment processing, TFT forecasting experiments, and the dashboard have been implemented. Forecasting performance and timestamp alignment require further validation. The prototype is not a production trading system and does not provide validated trading signals.

Final documentation and the demonstration video are the remaining submission deliverables.