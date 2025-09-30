import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def setup_plotting():
    """Setup plotting style and parameters"""
    plt.style.use('seaborn-v0_8')
    plt.rcParams['figure.figsize'] = (10, 4)
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.titlesize'] = 11
    plt.rcParams['axes.labelsize'] = 9

def print_section_header(title):
    """Print formatted section header"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def format_currency(amount):
    """Format amount as currency"""
    return f"${amount:,.2f}"

def print_progress(step, total_steps, message):
    """Print progress message"""
    print(f"\n[{step}/{total_steps}] {message}")