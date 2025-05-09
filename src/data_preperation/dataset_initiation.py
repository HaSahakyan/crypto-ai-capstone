import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '...', 'data_preperation')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'feature_engineering')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'simple_trend'))) 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_time_grouping')))

import pandas as pd
from data_time_grouping import CandleAggregator
from feature_engineering import Data_Featuring
from simple_trend import TargetCreatorSimple



def prepare_dataset(
    df,
    aggregate=False,
    group_length=5,
    feature_class=Data_Featuring,
    target_class=TargetCreatorSimple,
    target_params=None,
    trend_params=None,
    visualize_trends=False,
    join_df=[],                # NEW: DataFrame to join with
    join_on='date',             # NEW: join key(s)
    join_how='left'             # Optional: join type
):

    # Step 1: Aggregate if needed
    if aggregate:
        aggregator = CandleAggregator(group_length=group_length)
        df = aggregator.aggregate(df)

    # Step 2: Feature engineering
    fe = feature_class(df)
    fe.all_features()
    df_feat = fe.get_df()

    # Step 2.5: Optional join after feature engineering
    if join_df is not []:
        for fg in join_df:
            df['date'] = pd.to_datetime(df['date'])
            df_feat = pd.merge(df_feat, fg, on=join_on, how=join_how)

    # Step 3: Trend detection & labeling
    target_params = target_params or {
        'price_type': 'close',
        'date_col': 'datetime',
        'pct_threshold': 0.0035,
        'min_length': 3,
        'smoothing_factor': 0.001
    }
    trend_params = trend_params or {
        'flactuation_length': 5,
        'flactuation_threshold': 0.01
    }

    creator = target_class(df_feat, **target_params)
    creator.detect_trends(**trend_params)
    creator.add_trend_columns(direction=['Upward', 'Downward'])
    if visualize_trends:
        creator.visualize_trends()
    df_final = creator.get_df(reset_index=True)

    return df_final
