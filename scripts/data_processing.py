import pandas as pd
import numpy as np

def ip_to_int(ip):
    """Convert IP string to integer for range-based lookup."""
    try:
        parts = list(map(int, ip.split('.')))
        return (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]
    except:
        return 0

def merge_geolocation(fraud_df, ip_df):
    """Merge fraud data with country data using IP ranges."""
    # Convert IP addresses to integer format [cite: 128]
    fraud_df['ip_address_int'] = fraud_df['ip_address'].apply(ip_to_int)
    
    # Sort for merge_asof
    fraud_df = fraud_df.sort_values('ip_address_int')
    ip_df = ip_df.sort_values('lower_bound_ip_address')
    
    # Range-based lookup [cite: 129]
    merged_df = pd.merge_asof(
        fraud_df, ip_df, 
        left_on='ip_address_int', 
        right_on='lower_bound_ip_address'
    )
    
    # Validate if IP is within the upper bound
    merged_df['country'] = np.where(
        merged_df['ip_address_int'] <= merged_df['upper_bound_ip_address'], 
        merged_df['country'], 
        'Unknown'
    )
    return merged_df