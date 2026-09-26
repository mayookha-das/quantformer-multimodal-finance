
def extract_orderbook_features(order_book):
    """Calculate features from a five-level order book."""

    bids = order_book["bids"]
    asks = order_book["asks"]

    # Best available bid and ask prices
    best_bid = bids[0]["price"]
    best_ask = asks[0]["price"]

    # Difference between best ask and best bid
    spread = round(best_ask - best_bid, 4)

    # Midpoint between best bid and best ask
    mid_price = round((best_bid + best_ask) / 2, 4)

    # Total volume across all five price levels
    total_bid_volume = sum(level["volume"] for level in bids)
    total_ask_volume = sum(level["volume"] for level in asks)

    # Order-book imbalance
    total_volume = total_bid_volume + total_ask_volume

    if total_volume > 0:
        imbalance = (
            (total_bid_volume - total_ask_volume) / total_volume
        )
    else:
        imbalance = 0.0

    features = {
        "timestamp": order_book["timestamp"],
        "symbol": order_book["symbol"],
        "mid_price": mid_price,
        "spread": spread,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "total_bid_volume": total_bid_volume,
        "total_ask_volume": total_ask_volume,
        "orderbook_imbalance": round(imbalance, 4),
    }

    return features