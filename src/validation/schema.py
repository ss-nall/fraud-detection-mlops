COLUMN_SCHEMA = {

    "transaction_id": {
        "dtype": "object"
    },

    "timestamp": {
        "dtype": "object"
    },

    "sender_id": {
        "dtype": "object"
    },

    "receiver_id": {
        "dtype": "object"
    },

    "amount": {
        "dtype": ["float64", "int64"],
        "min": 0
    },

    "transaction_type": {
        "dtype": "object"
    },

    "sender_balance_before": {
        "dtype": ["float64", "int64"],
        "min": 0
    },

    "sender_balance_after": {
        "dtype": ["float64", "int64"],
        "min": 0
    },

    "receiver_balance_before": {
        "dtype": ["float64", "int64"],
        "min": 0
    },

    "receiver_balance_after": {
        "dtype": ["float64", "int64"],
        "min": 0
    },

    "device_id": {
        "dtype": "object"
    },

    "location_lat": {
        "dtype": ["float64", "int64"],
        "min": -90,
        "max": 90
    },

    "location_lon": {
        "dtype": ["float64", "int64"],
        "min": -180,
        "max": 180
    },

    "is_fraud": {
        "dtype": ["int64", "int32"],
        "allowed": [0, 1]
    }
}