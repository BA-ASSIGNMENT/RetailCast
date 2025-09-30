# Configuration and constants
import warnings
warnings.filterwarnings('ignore')

# File paths
DATA_PATH = 'dataset/online_retail.csv'

# Model parameters
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 7)
FORECAST_STEPS = 90
CLUSTER_RANGE = range(2, 8)
ENCODING_DIM = 32
TOP_N_PRODUCTS = 500
N_COMPONENTS_SVD = 50

# NLP Parameters
POSITIVE_TEMPLATES = [
    "Excellent quality {product}! Highly recommend.",
    "Love this {product}, exactly what I needed.",
    "Great {product}, fast delivery and good price.",
    "Amazing {product}! Will definitely buy again.",
    "Perfect {product}, exceeded my expectations.",
    "Fantastic quality {product}, very satisfied.",
    "Best {product} I've purchased, great value.",
    "Outstanding {product}, exactly as described.",
]

NEGATIVE_TEMPLATES = [
    "Disappointed with this {product}, poor quality.",
    "Not happy with the {product}, doesn't work well.",
    "Bad experience with this {product}, would not recommend.",
    "Poor quality {product}, waste of money.",
    "Terrible {product}, arrived damaged.",
    "Not satisfied with this {product}, returning it.",
    "Low quality {product}, not worth the price.",
    "Avoid this {product}, very disappointing.",
]

NEUTRAL_TEMPLATES = [
    "The {product} is okay, nothing special.",
    "Average {product}, does the job.",
    "Decent {product}, meets basic expectations.",
    "The {product} is fine, as expected.",
]

# Optimized Visualization settings
PLOT_STYLE = 'seaborn-v0_8'
FIGURE_SIZE = (10, 4)  # Reduced from (14, 6)
DPI = 150  # Reduced from 300
COMPACT_FIGURE_SIZE = (8, 3)  # For simpler plots


# Add to existing config.py

# Chatbot & Speech Settings
CHATBOT_TIMEOUT = 8  # seconds
SPEECH_RATE = 150    # words per minute for TTS
MAX_CONVERSATION_HISTORY = 100