# ============================================
# MODEL CONFIGURATION
# ============================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

SMOTE_RANDOM_STATE = 42

# ============================================
# THRESHOLD TUNING
# ============================================

THRESHOLD_START = 0.10

THRESHOLD_END = 0.95

THRESHOLD_STEP = 0.01

# ============================================
# FRAUD RULES
# ============================================

HIGH_VALUE_AMOUNT = 100000

RAPID_TRANSACTION_SECONDS = 60

# ============================================
# KENYA GEO BOUNDS
# ============================================

KENYA_MIN_LAT = -4.7
KENYA_MAX_LAT = 5.5

KENYA_MIN_LON = 33.5
KENYA_MAX_LON = 42.0

N_ESTIMATORS = 300

LEARNING_RATE = 0.05

NUM_LEAVES = 31

MAX_DEPTH = -1