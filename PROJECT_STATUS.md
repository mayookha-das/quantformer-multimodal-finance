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