import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import io
import plotly.express as px

# --- 1. Constants and Mappings ---

# Underwriting Class Mapping
UW_CLASS_MAPPING = {
    'W12GI': 'GI', 'W12GI FL': 'GI', 'W12GI KS': 'GI', 'W12GI KY': 'GI', 'W12GI MO': 'GI',
    'W12GI MT': 'GI', 'W12GI PA': 'GI', 'W12GI SC': 'GI', 'W12GI TX': 'GI', 'W12GI WI': 'GI',
    'W12GI WY': 'GI', 'WL12G': 'Graded', 'WL12G FL': 'Graded', 'WL12G KS': 'Graded',
    'WL12G KY': 'Graded', 'WL12G MO': 'Graded', 'WL12G MT': 'Graded', 'WL12G PA': 'Graded',
    'WL12G SC': 'Graded', 'WL12G TX': 'Graded', 'WL12G WI': 'WI', 'WL12G WY': 'Graded',
    'WL12P': 'Preferred', 'WL12P FL': 'Preferred', 'WL12P KS': 'Preferred', 'WL12P KY': 'Preferred',
    'WL12P MO': 'Preferred', 'WL12P MT': 'Preferred', 'WL12P PA': 'Preferred', 'WL12P SC': 'Preferred',
    'WL12P TX': 'Preferred', 'WL12P WI': 'Preferred', 'WL12P WY': 'Preferred', 'WL12S': 'Standard',
    'WL12S FL': 'Standard', 'WL12S KS': 'Standard', 'WL12S KY': 'Standard', 'WL12S MO': 'Standard',
    'WL12S MT': 'Standard', 'WL12S PA': 'Standard', 'WL12S SC': 'Standard', 'WL12S TX': 'Standard',
    'WL12S WI': 'Standard', 'WL12S WY': 'Standard'
}

# Commission Factor Mapping
FACTOR_PRE_JUL_2025 = {"Preferred": 1.5, "Standard": 1.5, "Graded": 1.15, "GI": 0.9}
FACTOR_POST_JUL_2025 = {"Preferred": 1.525, "Standard": 1.525, "Graded": 1.2, "GI": 0.975}
JUL_2025_CUTOFF = datetime(2025, 7, 1)

# Base Survival Curves (P(survive up to month m)) by UW Class
# These are the default values presented in the UI.
BASE_SURVIVAL_CURVES = {
    "Preferred": {1: 0.60, 2: 0.50, 3: 0.45, 4: 0.40, 5: 0.35, 6: 0.30, 7: 0.29, 8: 0.28, 9: 0.27, 10: 0.26, 11: 0.25, 12: 0.24},
    "Standard": {1: 0.60, 2: 0.50, 3: 0.45, 4: 0.40, 5: 0.35, 6: 0.30, 7: 0.29, 8: 0.28, 9: 0.27, 10: 0.26, 11: 0.25, 12: 0.24},
    "Graded": {1: 0.55, 2: 0.45, 3: 0.40, 4: 0.35, 5: 0.30, 6: 0.25, 7: 0.24, 8: 0.23, 9: 0.22, 10: 0.21, 11: 0.20, 12: 0.18},
    "GI": {1: 0.45, 2: 0.35, 3: 0.30, 4: 0.25, 5: 0.20, 6: 0.15, 7: 0.14, 8: 0.13, 9: 0.12, 10: 0.11, 11:0.10, 12: 0.08},
}

# Future Policy Distribution
UW_CLASS_DISTRIBUTION = {
    "Preferred": 0.18, "Standard": 0.50, "Graded": 0.20, "GI": 0.12
}

# Simulation Period
SIM_START_DATE = datetime(2025, 11, 1) 
SIM_END_DATE = datetime(2028, 1, 1)
HISTORICAL_CUTOFF_DATE = SIM_START_DATE - timedelta(days=1) # Oct 31, 2025
PROJECTION_DISPLAY_START = datetime(2024, 11, 1)  # Start charts 1 year before simulation

# --- 2. Helper Functions ---

def parse_date(date_str):
    """Parses date from YYYYMMDD format."""
    if pd.isna(date_str) or date_str == '':
        return None
    try:
        return datetime.strptime(str(int(date_str)), '%Y%m%d')
    except (ValueError, TypeError):
        return None

def calculate_commission_factor(uw_class, issue_date):
    """Determines CommissionFactor based on IssueDate."""
    if issue_date < JUL_2025_CUTOFF:
        return FACTOR_PRE_JUL_2025.get(uw_class, 0.0)
    else:
        return FACTOR_POST_JUL_2025.get(uw_class, 0.0)

def calculate_months_diff(d1, d2):
    """
    Calculates the difference in full months between two dates (d2 - d1).
    Used to determine months survived (full months active).
    """
    if pd.isna(d1) or pd.isna(d2) or d2 < d1:
        return 0
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)

def simulate_lapse_month(months_paid_to_date, survival_curve):
    """
    Simulates the estimated lapse month (m, where 1-12) based on conditional lapse probabilities
    derived from the cumulative survival curve.
    
    IMPORTANT: If lapse occurs in month m, the policy survived (m-1) complete months.
    Example: lapse_month=3 means the policy survived months 1 and 2, then lapsed during month 3.
    
    Logic:
    - Conditional lapse q(t) = 1 - [S(t) / S(t-1)] where S(t) is cumulative survival
    - A policy survives month t with probability p(t) = S(t) / S(t-1)
    - A policy lapses in month t with probability q(t) = 1 - p(t)
    
    Returns: 
        - lapse_month: The month number (1-12) in which lapse occurs, or 13 if policy survives all 12 months
        - paidm_indicators: Dict with keys 'Paidm1' through 'Paidm12' and values 0 or 1
    """
    
    conditional_lapse_probs = {}
    prev_survival = 1.0
    
    # Validate and warn about non-monotonic curves
    non_monotonic_warning = []
    
    for m in range(1, 13):
        # Get cumulative survival S(t)
        current_survival = survival_curve.get(m, survival_curve.get(12, 0.0))
        
        # Check for non-monotonic input
        if current_survival > prev_survival:
            non_monotonic_warning.append(f"M{m}: {current_survival} > M{m-1}: {prev_survival}")
        
        # Ensure survival probabilities are monotonically decreasing
        current_survival = max(0, min(current_survival, prev_survival))
        
        # Calculate conditional lapse probability q(t) = 1 - [S(t) / S(t-1)]
        # This is the probability of lapsing IN month m, given survival through month m-1
        if prev_survival > 0:
            conditional_survival_pt = current_survival / prev_survival
            conditional_lapse_qt = 1.0 - conditional_survival_pt
        else:
            conditional_lapse_qt = 1.0  # If S(t-1) = 0, policy has already lapsed
        
        conditional_lapse_probs[m] = conditional_lapse_qt
        prev_survival = current_survival
    
    # Display warning if curves were adjusted (only once per simulation)
    if non_monotonic_warning and not hasattr(simulate_lapse_month, '_warning_shown'):
        st.warning(f"⚠️ Survival curve was adjusted to be monotonically decreasing. Non-monotonic values detected at: {', '.join(non_monotonic_warning[:3])}")
        simulate_lapse_month._warning_shown = True

    # If the policy has already survived some months, we need to simulate
    # which month it will lapse in, starting from month (months_paid_to_date + 1)
    start_month = months_paid_to_date + 1
    
    # Initialize PaidmX indicators - months already paid are set to 1
    paidm_indicators = {}
    for i in range(1, 13):
        if i <= months_paid_to_date:
            paidm_indicators[f'Paidm{i}'] = 1
        else:
            paidm_indicators[f'Paidm{i}'] = 0
    
    if start_month > 12:
        # Policy already survived 12 months, all PaidmX = 1
        for i in range(1, 13):
            paidm_indicators[f'Paidm{i}'] = 1
        return 13, paidm_indicators

    # Simulate month-by-month using conditional probabilities (Bernoulli trials)
    # Starting from start_month, check each month if the policy lapses
    for m in range(start_month, 13):
        # Get the conditional lapse probability for this month
        lapse_prob = conditional_lapse_probs[m]
        
        # Draw a random number to see if policy lapses this month
        if np.random.random() < lapse_prob:
            # Policy lapses in month m
            # It survived months 1 through (m-1), so those are paid
            for i in range(1, m):
                paidm_indicators[f'Paidm{i}'] = 1
            # Month m and beyond are NOT paid (already set to 0)
            return m, paidm_indicators
        else:
            # Policy survived month m
            paidm_indicators[f'Paidm{m}'] = 1
    
    # If we get here, policy survived all 12 months
    for i in range(1, 13):
        paidm_indicators[f'Paidm{i}'] = 1
    return 13, paidm_indicators

# --- 3. Data Processing and Modeling ---

def adjust_lapse_date_for_month_end_bias(est_lapse_date, issue_date):
    """
    Adjusts the estimated lapse date to model the carrier's behavior where term dates
    are disproportionately set at the end or very beginning of the month.
    
    This creates a concentration of chargebacks in the first full week of each month.
    
    Args:
        est_lapse_date: The originally calculated lapse date
        issue_date: The policy issue date
        
    Returns:
        Adjusted lapse date that's biased toward month boundaries
    """
    if pd.isna(est_lapse_date):
        return est_lapse_date
    
    # 50% chance to move to month boundary (end of month or start of next)
    if np.random.random() < 0.5:
        # Get the month of the original lapse date
        lapse_month = est_lapse_date.month
        lapse_year = est_lapse_date.year
        
        # 50/50 split between end of current month vs. start of next month
        if np.random.random() < 0.5:
            # Move to end of the current month
            if lapse_month == 12:
                next_month = 1
                next_year = lapse_year + 1
            else:
                next_month = lapse_month + 1
                next_year = lapse_year
            
            # Last day of current month = day before first day of next month
            adjusted_date = datetime(next_year, next_month, 1) - timedelta(days=1)
        else:
            # Move to start of the current month (1st of the month)
            adjusted_date = datetime(lapse_year, lapse_month, 1)
        
        # Don't move the date earlier than the issue date
        if adjusted_date < issue_date:
            return est_lapse_date
        
        return adjusted_date
    else:
        # Leave date as-is (50% of the time, normal distribution throughout month)
        return est_lapse_date


def load_and_process_data(uploaded_file, custom_survival_curves):
    """
    Loads, cleans, filters, and calculates initial fields for the historical book.
    Also handles lapse simulation for active policies using class-specific curves.
    """
    if uploaded_file is None:
        st.error("Please upload the gtlpolicies.csv file.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(uploaded_file, on_bad_lines='skip')
    except Exception as e:
        st.error(f"Failed to read CSV file: {e}")
        return pd.DataFrame()
        
    initial_rows = len(df)
    st.info(f"Loaded {initial_rows} policies from CSV (skipping any malformed lines).")

    # CRITICAL: Strip whitespace from plan_code
    if 'plan_code' in df.columns:
        df['plan_code'] = df['plan_code'].astype(str).str.strip()

    # 1. Filter and Map
    valid_plan_codes = list(UW_CLASS_MAPPING.keys())
    df = df[df['plan_code'].isin(valid_plan_codes)].copy()
    if 'plan_code' not in df.columns:
        st.error("The uploaded CSV must contain a 'plan_code' column.")
        return pd.DataFrame()

    df['UWClass'] = df['plan_code'].map(UW_CLASS_MAPPING)

    # 2. Clean and Standardize Columns
    df = df.rename(columns={'policy_nbr': 'PolicyNumber', 'annual_premium': 'AnnualPremium'})
    df['IssueDate'] = df['Issue Date'].apply(parse_date)
    # IMPORTANT: ApplicationDate is set from app_recvd_date (the date the advance was received)
    df['ApplicationDate'] = df.get('app_recvd_date', pd.Series(dtype='datetime64[ns]')).apply(parse_date)
    # If app_recvd_date is missing or invalid, default ApplicationDate to IssueDate for historical policies
    df['ApplicationDate'] = df.apply(lambda row: row['IssueDate'] if pd.isna(row['ApplicationDate']) else row['ApplicationDate'], axis=1)
    
    df['paid_to_date'] = df['paid_to_date'].apply(parse_date)
    df['TermDate_Source'] = df['Term Date'].apply(parse_date) 
    
    required_cols = ['PolicyNumber', 'AnnualPremium', 'UWClass', 'IssueDate', 'ApplicationDate', 'paid_to_date', 'TermDate_Source']
    
    df = df.dropna(subset=['IssueDate', 'ApplicationDate']) # Filter out records where IssueDate or ApplicationDate is missing
    df = df.reindex(columns=required_cols) # Ensure column order/existence

    df['PolicyNumber'] = df['PolicyNumber'].astype(str)

    if df.empty:
        return df

    # 3. Calculate Commission Factor and Advance
    df['CommissionFactor'] = df.apply(
        lambda row: calculate_commission_factor(row['UWClass'], row['IssueDate']), axis=1
    )
    df['Advance'] = df['AnnualPremium'] * df['CommissionFactor']
    
    # Initialize projection columns
    df['TermDate'] = df['TermDate_Source']
    df['EstLapseDate'] = None
    df['ActualMonthsSurvived'] = 0  # NEW: Track actual months survived
    df['EarnedCommission'] = 0.0
    df['UnearnedCommission'] = df['Advance']
    df['Status'] = 'Active' 

    for i in range(1, 13):
        df[f'Paidm{i}'] = 0
        
    # --- 4. Historical Modeling (up to HISTORICAL_CUTOFF_DATE) ---
    
    def model_policy(row):
        # Determine the survival curve based on the policy's UWClass
        survival_curve = custom_survival_curves.get(row['UWClass'], BASE_SURVIVAL_CURVES['Standard']) 

        is_terminated_historically = pd.notna(row['TermDate_Source']) and row['TermDate_Source'] <= HISTORICAL_CUTOFF_DATE
        is_active_historically = row['IssueDate'] <= HISTORICAL_CUTOFF_DATE and not is_terminated_historically

        # Initialize tracking variables
        actual_months_survived = 0
        simulated_months_survived = 0

        if is_terminated_historically:
            # This policy actually lapsed - use real data
            lapse_date = row['TermDate_Source']
            actual_months_survived = calculate_months_diff(row['IssueDate'], lapse_date)
            
            row['EstLapseDate'] = lapse_date 
            row['TermDate'] = lapse_date
            row['Status'] = 'Lapsed'
            row['ActualMonthsSurvived'] = actual_months_survived
            
            # For lapsed policies, PaidmX reflects actual survival
            simulated_months_survived = actual_months_survived
            
            # Set PaidmX based on actual months survived
            for m in range(1, 13):
                if m <= actual_months_survived:
                    row[f'Paidm{m}'] = 1
                else:
                    row[f'Paidm{m}'] = 0
            
        elif is_active_historically: 
            # This policy is still active - simulate future lapse
            
            # Calculate how many months the policy has been paid through
            effective_paid_to_date = min(row['paid_to_date'], HISTORICAL_CUTOFF_DATE) if pd.notna(row['paid_to_date']) else row['IssueDate']
            actual_months_survived = calculate_months_diff(row['IssueDate'], effective_paid_to_date)
            actual_months_survived = min(actual_months_survived, 12)
            
            row['ActualMonthsSurvived'] = actual_months_survived
            
            # Simulate when this policy will lapse using Bernoulli trials
            lapse_month_number, paidm_indicators = simulate_lapse_month(actual_months_survived, survival_curve)
            
            # Set PaidmX from Bernoulli trial results
            for m in range(1, 13):
                row[f'Paidm{m}'] = paidm_indicators[f'Paidm{m}']
            
            if lapse_month_number <= 12:
                # CORRECTED: If lapse in month m, survived (m-1) complete months
                # Lapse date is set to somewhere in month m
                # NOTE: Do NOT apply month-end bias to historical policies - they will
                # naturally exhibit the carrier's behavior when they actually lapse
                row['EstLapseDate'] = row['IssueDate'] + relativedelta(months=lapse_month_number - 1)
                simulated_months_survived = lapse_month_number - 1
            else:
                # Policy survives all 12 months
                row['EstLapseDate'] = SIM_END_DATE + timedelta(days=1)
                simulated_months_survived = 12 
                
            row['TermDate'] = None 
            row['Status'] = 'Active'
            
        else:
            # Future Policy (IssueDate > Cutoff), shouldn't be in historical set
            actual_months_survived = 0
            row['ActualMonthsSurvived'] = 0
            
            lapse_month_number, paidm_indicators = simulate_lapse_month(0, survival_curve)
            
            # Set PaidmX from Bernoulli trial results
            for m in range(1, 13):
                row[f'Paidm{m}'] = paidm_indicators[f'Paidm{m}']
            
            if lapse_month_number <= 12:
                # CORRECTED: If lapse in month m, survived (m-1) complete months
                # NOTE: This branch shouldn't occur in historical data processing,
                # but keep it consistent - no bias here
                row['EstLapseDate'] = row['IssueDate'] + relativedelta(months=lapse_month_number - 1)
                simulated_months_survived = lapse_month_number - 1
            else:
                row['EstLapseDate'] = SIM_END_DATE + timedelta(days=1)
                simulated_months_survived = 12 
                
            row['TermDate'] = None 
            row['Status'] = 'Future Active'

        # CORRECTED: Commission Earning uses actual months survived (for historical accuracy)
        # This represents what has actually been earned based on real payment history
        months_for_earned_calc = actual_months_survived
        earned_fraction = months_for_earned_calc / 12.0
        earned_fraction = min(earned_fraction, 1.0)
        
        row['EarnedCommission'] = row['Advance'] * earned_fraction
        row['UnearnedCommission'] = row['Advance'] - row['EarnedCommission']

        return row

    df = df.apply(model_policy, axis=1)
    
    # Filter historical policies to only include those that are truly historical (IssueDate <= Cutoff)
    df_historical = df[df['IssueDate'] <= HISTORICAL_CUTOFF_DATE].copy()
    
    df_historical = df_historical.drop(columns=['TermDate_Source', 'paid_to_date'])
    df_historical['EstLapseDate'] = pd.to_datetime(df_historical['EstLapseDate'])

    st.success("Historical book loaded, modeled, and active policies have estimated lapse dates.")
    return df_historical

# --- 5. Future Policy Generation ---

def generate_future_policies(historical_df, initial_policies, weekly_change, ceiling, volume_randomness, custom_survival_curves):
    """
    Generates new simulated policies with variable volume based on input parameters,
    using class-specific survival curves for lapse simulation.
    
    Args:
        historical_df: Historical policies dataframe
        initial_policies: Starting policies per week
        weekly_change: Incremental change per week
        ceiling: Maximum policies per week
        volume_randomness: Percentage variation (0-50) in weekly volume
        custom_survival_curves: Survival curves by UW class
    """
    date_range = pd.date_range(start=SIM_START_DATE, end=SIM_END_DATE, freq='W')
    
    new_policies = []
    premium_pool = historical_df['AnnualPremium'].tolist()
    
    if not premium_pool:
        st.warning("No valid historical policies to sample premiums from. Skipping future generation.")
        return pd.DataFrame()
        
    classes = list(UW_CLASS_DISTRIBUTION.keys())
    probs = list(UW_CLASS_DISTRIBUTION.values())
    
    current_policies_per_week = initial_policies
    
    for i, week_end_date in enumerate(date_range):
        
        # 1. Determine the base number of policies for this week
        base_policies = current_policies_per_week
        
        # 2. Apply randomness
        if volume_randomness > 0:
            # Generate random variation within ±volume_randomness%
            variation_factor = 1.0 + np.random.uniform(-volume_randomness/100, volume_randomness/100)
            policies_this_week = int(round(base_policies * variation_factor))
        else:
            policies_this_week = int(round(base_policies))
        
        # Ensure we respect the ceiling
        policies_this_week = min(policies_this_week, ceiling)
        policies_this_week = max(0, policies_this_week)
        
        # 3. Update volume for the next week
        current_policies_per_week += weekly_change
        current_policies_per_week = min(current_policies_per_week, ceiling)
        current_policies_per_week = max(0, current_policies_per_week)

        week_start_date = week_end_date - timedelta(days=6)
        
        for j in range(policies_this_week):
            policy_id = f"SIM_{len(new_policies) + 1}"
            
            random_offset = np.random.randint(0, 7)
            application_date = week_start_date + timedelta(days=random_offset)
            
            # CORRECTED: Use IssueDate same as ApplicationDate for consistency
            # Advances are paid on application, commission earning starts on issue
            issue_date = application_date  # Changed from week_end_date
            
            annual_premium = np.random.choice(premium_pool)
            uw_class = np.random.choice(classes, p=probs)
            
            comm_factor = calculate_commission_factor(uw_class, issue_date)
            advance = annual_premium * comm_factor
            
            # Determine the survival curve for the class
            survival_curve = custom_survival_curves.get(uw_class, BASE_SURVIVAL_CURVES['Standard'])
            
            # Simulate lapse month 'm' using Bernoulli trials
            lapse_month_number, paidm_indicators = simulate_lapse_month(0, survival_curve)
            
            # CORRECTED: If lapse in month m, survived (m-1) complete months
            if lapse_month_number <= 12:
                initial_lapse_date = issue_date + relativedelta(months=lapse_month_number - 1)
                # Apply month-end bias for simulated data
                est_lapse_date = adjust_lapse_date_for_month_end_bias(initial_lapse_date, issue_date)
                months_survived = lapse_month_number - 1
            else:
                est_lapse_date = SIM_END_DATE + timedelta(days=1)
                months_survived = 12 
                
            policy = {
                'PolicyNumber': policy_id,
                'AnnualPremium': annual_premium,
                'UWClass': uw_class,
                'IssueDate': issue_date,
                'ApplicationDate': application_date, 
                'CommissionFactor': comm_factor,
                'Advance': advance,
                'TermDate': None,
                'EstLapseDate': est_lapse_date,
                'ActualMonthsSurvived': 0,  # Future policies haven't survived yet
                'EarnedCommission': 0.0,
                'UnearnedCommission': advance,
                'Status': 'Future Active',
            }
            
            # Set PaidmX based on Bernoulli trial results
            for k in range(1, 13):
                policy[f'Paidm{k}'] = paidm_indicators[f'Paidm{k}']

            new_policies.append(policy)

    new_df = pd.DataFrame(new_policies)
    st.success(f"Generated {len(new_policies)} new simulated policies with variable volume (Avg: {len(new_policies)/len(date_range):.0f}/week).")
    return new_df


# --- 6. Cash Flow Projection Engine ---

def project_cash_flow(all_policies_df):
    """
    Projects weekly cash flow (Advances, Earned Commission, Chargebacks) 
    from PROJECTION_DISPLAY_START to SIM_END_DATE.
    
    CORRECTED: Now clearly separates historical (actual) vs projected (estimated) data.
    Note: Projection display starts 1 year before simulation to show historical context.
    """
    date_range = pd.date_range(start=PROJECTION_DISPLAY_START, end=SIM_END_DATE, freq='W')
    
    projection = {
        'WeekEndDate': [],
        'WeeklyAdvances': [],
        'WeeklyChargebacks': [],
        'WeeklyEarnedCommission': [],
        'WeeklyUnearnedCommission': [],
    }
    
    all_policies_df['DailyEarnRate'] = all_policies_df['Advance'] / 365
    
    for week_end_date in date_range:
        
        week_start_date = week_end_date - timedelta(days=6)
        
        # --- 1. Advances (Cash In) ---
        # Use ApplicationDate for when advances are received
        # For historical policies: ApplicationDate = app_recvd_date (actual advance date)
        # For future policies: ApplicationDate = simulated application date
        weekly_advances = all_policies_df[
            (all_policies_df['ApplicationDate'] >= week_start_date) & 
            (all_policies_df['ApplicationDate'] <= week_end_date)
        ]['Advance'].sum()
        
        # --- 2. Chargebacks (Cash Out) ---
        # For historical lapses: use actual TermDate
        # For projected lapses: use EstLapseDate
        
        # Historical lapses that occurred this week
        historical_chargebacks = all_policies_df[
            (all_policies_df['Status'] == 'Lapsed') &
            (all_policies_df['TermDate'] >= week_start_date) & 
            (all_policies_df['TermDate'] <= week_end_date)
        ].copy()
        
        if not historical_chargebacks.empty:
            historical_chargebacks['MonthsElapsed'] = historical_chargebacks['ActualMonthsSurvived']
            historical_chargebacks['MonthsElapsed'] = historical_chargebacks['MonthsElapsed'].clip(upper=12)
            historical_chargebacks['EarnedAtLapse'] = historical_chargebacks['Advance'] * (historical_chargebacks['MonthsElapsed'] / 12.0)
            historical_chargebacks['ChargebackAmount'] = historical_chargebacks['Advance'] - historical_chargebacks['EarnedAtLapse']
            historical_chargeback_amount = historical_chargebacks['ChargebackAmount'].sum()
        else:
            historical_chargeback_amount = 0.0
        
        # Projected lapses that will occur this week
        projected_chargebacks = all_policies_df[
            (all_policies_df['Status'].isin(['Active', 'Future Active'])) &
            (all_policies_df['EstLapseDate'] >= week_start_date) & 
            (all_policies_df['EstLapseDate'] <= week_end_date)
        ].copy()

        if not projected_chargebacks.empty:
            projected_chargebacks['MonthsElapsed'] = projected_chargebacks.apply(
                lambda row: calculate_months_diff(row['IssueDate'], row['EstLapseDate']), axis=1
            )
            projected_chargebacks['MonthsElapsed'] = projected_chargebacks['MonthsElapsed'].clip(upper=12)
            projected_chargebacks['EarnedAtLapse'] = projected_chargebacks['Advance'] * (projected_chargebacks['MonthsElapsed'] / 12.0)
            projected_chargebacks['ChargebackAmount'] = projected_chargebacks['Advance'] - projected_chargebacks['EarnedAtLapse']
            projected_chargeback_amount = projected_chargebacks['ChargebackAmount'].sum()
        else:
            projected_chargeback_amount = 0.0

        weekly_chargebacks = historical_chargeback_amount + projected_chargeback_amount

        # --- 3. Commission Earning (Accrual) ---
        # CORRECTED: Only earn on policies that are still active (not yet lapsed)
        # Also fix the issue where policies that lapse this week might still earn
        earning_policies = all_policies_df[
            (all_policies_df['IssueDate'] <= week_end_date) & 
            (
                # For lapsed policies: only earn if not yet lapsed
                ((all_policies_df['Status'] == 'Lapsed') & (all_policies_df['TermDate'] > week_end_date)) |
                # For active/future policies: only earn if EstLapseDate is after this week
                ((all_policies_df['Status'].isin(['Active', 'Future Active'])) & (all_policies_df['EstLapseDate'] > week_end_date))
            )
        ].copy()
        
        if not earning_policies.empty:
            earning_policies['DaysSinceIssue'] = (week_end_date - earning_policies['IssueDate']).dt.days
            earning_policies['TotalEarnedUpToDate'] = earning_policies['DailyEarnRate'] * earning_policies['DaysSinceIssue']
            earning_policies['TotalEarnedUpToDate'] = earning_policies['TotalEarnedUpToDate'].clip(upper=earning_policies['Advance'])

            prev_week_end_date = week_end_date - timedelta(days=7)
            earning_policies['DaysSinceIssuePrev'] = (prev_week_end_date - earning_policies['IssueDate']).dt.days.clip(lower=0)
            earning_policies['TotalEarnedUpToPrevDate'] = earning_policies['DailyEarnRate'] * earning_policies['DaysSinceIssuePrev']
            earning_policies['TotalEarnedUpToPrevDate'] = earning_policies['TotalEarnedUpToPrevDate'].clip(upper=earning_policies['Advance'])
            
            weekly_earned = (earning_policies['TotalEarnedUpToDate'] - earning_policies['TotalEarnedUpToPrevDate']).clip(lower=0).sum()
        else:
            weekly_earned = 0.0
        
        # --- 4. Total Unearned Commission (Snapshot) ---
        snapshot_policies = all_policies_df[
            (all_policies_df['IssueDate'] <= week_end_date) & 
            (
                # For lapsed: only count if not yet lapsed
                ((all_policies_df['Status'] == 'Lapsed') & (all_policies_df['TermDate'] > week_end_date)) |
                # For active/future: only count if EstLapseDate is in the future
                ((all_policies_df['Status'].isin(['Active', 'Future Active'])) & (all_policies_df['EstLapseDate'] > week_end_date))
            )
        ].copy()

        if not snapshot_policies.empty:
            snapshot_policies['DaysActive'] = (week_end_date - snapshot_policies['IssueDate']).dt.days
            snapshot_policies['CurrentEarned'] = (snapshot_policies['DailyEarnRate'] * snapshot_policies['DaysActive']).clip(upper=snapshot_policies['Advance'])

            total_advances_active = snapshot_policies['Advance'].sum()
            total_earned_active = snapshot_policies['CurrentEarned'].sum()
            
            total_unearned_commission = total_advances_active - total_earned_active
        else:
            total_unearned_commission = 0.0

        projection['WeekEndDate'].append(week_end_date)
        projection['WeeklyAdvances'].append(weekly_advances)
        projection['WeeklyChargebacks'].append(weekly_chargebacks)
        projection['WeeklyEarnedCommission'].append(weekly_earned)
        projection['WeeklyUnearnedCommission'].append(total_unearned_commission)

    
    projection_df = pd.DataFrame(projection)
    
    projection_df['AccumulatedAdvances'] = projection_df['WeeklyAdvances'].cumsum()
    projection_df['AccumulatedChargebacks'] = projection_df['WeeklyChargebacks'].cumsum()
    projection_df['AccumulatedEarnedCommission'] = projection_df['WeeklyEarnedCommission'].cumsum()
    
    projection_df['WeeklyNetCashFlow'] = projection_df['WeeklyAdvances'] - projection_df['WeeklyChargebacks']
    projection_df['AccumulatedNetCashFlow'] = projection_df['WeeklyNetCashFlow'].cumsum()
    
    return projection_df

# --- 7. Survival Curve Analysis ---

def generate_volume_by_class(all_policies_df):
    """
    Generates weekly policy volume data grouped by underwriting class.
    Returns data suitable for a stacked bar chart.
    
    Uses ApplicationDate (app_recvd_date for historical policies) to show when
    advances were actually received, matching cash flow timing.
    """
    df_volume = all_policies_df.copy()
    
    # Filter to policies with ApplicationDate after PROJECTION_DISPLAY_START
    df_volume = df_volume[df_volume['ApplicationDate'] >= PROJECTION_DISPLAY_START].copy()
    
    if df_volume.empty:
        return pd.DataFrame()
    
    # Create ApplicationWeek based on when advances were received
    df_volume['ApplicationWeek'] = df_volume['ApplicationDate'].dt.to_period('W-SAT').dt.start_time
    
    # Group by week and UW class, count policies
    volume_by_class = df_volume.groupby(['ApplicationWeek', 'UWClass']).size().reset_index(name='PolicyCount')
    
    # Pivot to wide format for stacked bar chart
    volume_pivot = volume_by_class.pivot(index='ApplicationWeek', columns='UWClass', values='PolicyCount').fillna(0)
    
    # Ensure all classes are present
    for uw_class in ['Preferred', 'Standard', 'Graded', 'GI']:
        if uw_class not in volume_pivot.columns:
            volume_pivot[uw_class] = 0
    
    # Reorder columns for consistent stacking
    volume_pivot = volume_pivot[['Preferred', 'Standard', 'Graded', 'GI']]
    
    return volume_pivot.reset_index()


def generate_survival_curves(all_policies_df):
    """
    Calculates the 12-month survival curve for all policies (historical and simulated), 
    grouped by their IssueDate week.
    
    Now uses actual PaidmX indicators from Bernoulli trials for accurate survival representation.
    """
    df_survival = all_policies_df.copy()
    
    # Filter to policies issued after PROJECTION_DISPLAY_START for relevant cohort analysis
    df_survival = df_survival[df_survival['IssueDate'] >= PROJECTION_DISPLAY_START].copy()
    
    if df_survival.empty:
        return pd.DataFrame()
    
    df_survival['IssueWeek'] = df_survival['IssueDate'].dt.to_period('W-SAT').dt.start_time
    
    cohort_counts = df_survival.groupby('IssueWeek')['PolicyNumber'].count().rename('TotalPolicies')
    
    survival_data = []
    
    # For each month, calculate the survival rate based on PaidmX columns
    # PaidmX = 1 means the policy paid/survived through month X
    for m in range(1, 13):
        # Calculate survival rate as the proportion of policies with PaidmX = 1
        # This directly reflects the Bernoulli trial outcomes
        survived_rate_by_paidm = df_survival.groupby('IssueWeek')[f'Paidm{m}'].mean().rename(f'Survival_M{m}')
        survival_data.append(survived_rate_by_paidm)

    survival_df = pd.concat(survival_data, axis=1).reset_index()
    
    survival_df = survival_df.merge(cohort_counts.reset_index(), on='IssueWeek')

    # Reshape for Plotly chart 
    melted_df = survival_df.melt(
        id_vars=['IssueWeek', 'TotalPolicies'],
        value_vars=[f'Survival_M{m}' for m in range(1, 13)],
        var_name='Month',
        value_name='SurvivalRate'
    )
    
    melted_df['MonthNumber'] = melted_df['Month'].str.replace('Survival_M', '').astype(int)
    melted_df['IssueWeekStr'] = melted_df['IssueWeek'].dt.strftime('%Y-%m-%d')
    
    return melted_df

# --- 8. Streamlit UI ---

def calculate_conditional_probabilities(survival_data):
    """
    Calculate conditional survival and lapse probabilities from cumulative survival data.
    
    Returns a DataFrame with conditional probabilities for display.
    """
    data = []
    prev_survival = 1.0
    
    for month in range(1, 13):
        current_survival = survival_data.get(month, 0.0)
        
        # Ensure monotonic (clamp if needed)
        current_survival = max(0, min(current_survival, prev_survival))
        
        # Conditional survival: p(t) = S(t) / S(t-1)
        if prev_survival > 0:
            conditional_survival = current_survival / prev_survival
        else:
            conditional_survival = 0.0
            
        # Conditional lapse: q(t) = 1 - p(t)
        conditional_lapse = 1.0 - conditional_survival
        
        data.append({
            'Month': month,
            'Cumulative_Survival_St': current_survival,
            'Previous_Cumulative_St-1': prev_survival,
            'Conditional_Survival_pt': conditional_survival,
            'Conditional_Lapse_qt': conditional_lapse
        })
        
        prev_survival = current_survival
    
    return pd.DataFrame(data)


def render_survival_inputs(uw_class, defaults):
    """
    Renders 12 number inputs for a single UW class and returns the values,
    with live display of conditional probabilities.
    """
    st.subheader(f"{uw_class} Class Survival Probabilities")
    st.caption("Enter cumulative survival probabilities S(t) = P(policy survives through end of month t)")
    
    # Create input section
    st.markdown("**Input Cumulative Survival Rates:**")
    survival_data = {}
    cols = st.columns(6)
    
    for i in range(1, 13):
        col = cols[(i - 1) % 6]
        key = f"{uw_class}_M{i}_survival" 
        
        survival_data[i] = col.number_input(
            f"M{i}",
            min_value=0.0,
            max_value=1.0,
            value=defaults.get(i, 0.0), 
            step=0.01,
            format="%.2f",
            key=key,
            help=f"Cumulative probability of surviving through month {i}."
        )
    
    # Calculate and display conditional probabilities
    st.markdown("---")
    st.markdown("**Derived Conditional Probabilities:**")
    st.caption("Conditional Survival p(t) = probability of surviving month t given survival through month t-1")
    
    conditional_df = calculate_conditional_probabilities(survival_data)
    
    # Format the dataframe for display
    display_df = conditional_df.copy()
    display_df['Cumulative_Survival_St'] = display_df['Cumulative_Survival_St'].map('{:.2%}'.format)
    display_df['Previous_Cumulative_St-1'] = display_df['Previous_Cumulative_St-1'].map('{:.2%}'.format)
    display_df['Conditional_Survival_pt'] = display_df['Conditional_Survival_pt'].map('{:.4f}'.format)
    display_df['Conditional_Lapse_qt'] = display_df['Conditional_Lapse_qt'].map('{:.4f}'.format)
    
    display_df.columns = ['Month', 'Cumulative Survival S(t)', 'Previous S(t-1)', 'Conditional Survival p(t)', 'Conditional Lapse q(t)']
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    return survival_data


def main():
    st.set_page_config(layout="wide", page_title="Commission Cash Flow Projector")

    st.title("💰 Life Insurance Commission Cash Flow Projector")
    st.markdown("""
        Use this tool to project future cash flows, commission earnings, and chargeback risk.
        
        **Important Note:** "Lapse in month M" means the policy survived (M-1) complete months before lapsing.
    """)
    
    # Create tabs for better organization
    tab1, tab2 = st.tabs(["📊 Results & Visualization", "⚙️ Configuration & Inputs"])
    
    # ===== TAB 2: Configuration & Inputs =====
    with tab2:
        st.header("Configuration & Inputs")
        
        # File Upload Section
        st.subheader("📁 Data Upload")
        st.info("Using default policy data (randompolicydata). To use different data, upload a CSV file below.")
        uploaded_file = st.file_uploader(
            "Upload your gtlpolicies.csv file (optional - leave empty to use default data)", 
            type="csv",
            help="Requires 'policy_nbr', 'plan_code', 'annual_premium', 'Issue Date', 'paid_to_date', 'Term Date', and 'app_recvd_date'."
        )
        
        st.markdown("---")
        
        # Survival Curve Configuration
        st.subheader("📈 Lapse/Survival Rate Configuration")
        st.info("""
        Configure the cumulative survival probabilities for each underwriting class. 
        The conditional probabilities will be calculated and displayed automatically.
        """)
        
        custom_survival_curves = {}
        
        # Create expandable sections for each UW class
        with st.expander("🎯 Preferred Class", expanded=False):
            custom_survival_curves['Preferred'] = render_survival_inputs('Preferred', BASE_SURVIVAL_CURVES['Preferred'])
        
        with st.expander("📊 Standard Class", expanded=False):
            custom_survival_curves['Standard'] = render_survival_inputs('Standard', BASE_SURVIVAL_CURVES['Standard'])
        
        with st.expander("📉 Graded Class", expanded=False):
            custom_survival_curves['Graded'] = render_survival_inputs('Graded', BASE_SURVIVAL_CURVES['Graded'])
        
        with st.expander("⚡ GI Class", expanded=False):
            custom_survival_curves['GI'] = render_survival_inputs('GI', BASE_SURVIVAL_CURVES['GI'])
        
        st.markdown("---")
        
        # Volume Configuration
        st.subheader("📦 Future Policy Volume Configuration")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            initial_policies = st.number_input(
                f"Initial Policies per Week",
                min_value=0,
                value=10,
                step=10,
                key="initial_policies_input",
                help=f"Starting volume for week of {SIM_START_DATE.strftime('%b %d, %Y')}"
            )
        
        with col2:
            weekly_change = st.number_input(
                "Weekly Volume Change",
                value=5,
                step=1,
                key="weekly_change_input",
                help="Incremental change per week (can be negative for decreasing volume)"
            )
        
        with col3:
            ceiling = st.number_input(
                "Policy Volume Ceiling",
                min_value=0,
                value=100,
                step=10,
                key="ceiling_input",
                help="Maximum policies per week"
            )
        
        with col4:
            volume_randomness = st.number_input(
                "Volume Randomness (%)",
                min_value=0,
                max_value=50,
                value=10,
                step=5,
                key="volume_randomness_input",
                help="Percent variation in weekly volume (0-50%). E.g., 10% means volume can vary ±10% from target each week."
            )
        
        # Volume visualization
        st.markdown("**Volume Projection Preview:**")
        weeks = list(range(0, 12))  # First 12 weeks
        volumes = []
        current_vol = initial_policies
        for week in weeks:
            volumes.append(min(max(0, current_vol), ceiling))
            current_vol += weekly_change
        
        preview_df = pd.DataFrame({
            'Week': weeks,
            'Projected Policies': volumes
        })
        st.line_chart(preview_df.set_index('Week'))
        
        st.markdown("---")
        
        # Projection Dates
        st.subheader("📅 Projection Period")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Chart Display Start", PROJECTION_DISPLAY_START.strftime('%Y-%m-%d'))
        with col2:
            st.metric("Historical Cutoff", HISTORICAL_CUTOFF_DATE.strftime('%Y-%m-%d'))
        with col3:
            st.metric("Projection Start", SIM_START_DATE.strftime('%Y-%m-%d'))
        with col4:
            st.metric("Projection End", SIM_END_DATE.strftime('%Y-%m-%d'))
        
        st.markdown("---")
        
        # Modeling Notes
        st.subheader("📝 Modeling Notes")
        st.info("""
        **Chargeback Timing:** For simulated future policies (issued after Nov 2025), the model applies 
        the carrier's behavior where term dates are disproportionately set at month boundaries 
        (end of month or start of month). This creates a concentration of chargebacks in the first 
        full week of each month (~50% of monthly chargebacks), with the remainder distributed throughout 
        the month. Historical policies already exhibit this behavior naturally and are not adjusted.
        """)
        
        st.markdown("---")
        
        # Run Button
        run_simulation = st.button("▶️ Run Simulation", type="primary", use_container_width=True)
    
    # ===== TAB 1: Results & Visualization =====
    with tab1:
        st.header("Simulation Results")
        
        # Check if we should run the simulation
        if 'simulation_run' not in st.session_state:
            st.session_state.simulation_run = False
        
        if run_simulation:
            st.session_state.simulation_run = True
            st.session_state.uploaded_file = uploaded_file
            st.session_state.custom_survival_curves = custom_survival_curves
            st.session_state.initial_policies = initial_policies
            st.session_state.weekly_change = weekly_change
            st.session_state.ceiling = ceiling
            st.session_state.volume_randomness = volume_randomness
        
        if not st.session_state.simulation_run:
            st.info("👈 Configure your inputs in the 'Configuration & Inputs' tab, then click 'Run Simulation' to see results here.")
            st.markdown("### What you'll see after running the simulation:")
            st.markdown("""
            1. **Comprehensive Accumulated Metrics** - Track advances, chargebacks, earned/unearned commission, and net cash
            2. **Weekly Cash Flow Analysis** - See advances, chargebacks, and net flow
            3. **12-Month Policy Survival Curves** - Visualize cohort retention
            4. **Weekly Policy Volume by Underwriting Class** - Stacked bar chart showing new policy volumes
            5. **Average Weekly Cash Flows** - Moving averages to reveal trends
            6. **Downloadable Data** - Export full policy-level details
            """)
        else:
            # Retrieve stored values
            uploaded_file = st.session_state.uploaded_file
            custom_survival_curves = st.session_state.custom_survival_curves
            initial_policies = st.session_state.initial_policies
            weekly_change = st.session_state.weekly_change
            ceiling = st.session_state.ceiling
            volume_randomness = st.session_state.volume_randomness
            
            # If no file is uploaded, use the default randompolicydata file
            if uploaded_file is None:
                import os
                default_file_path = os.path.join(os.path.dirname(__file__), 'randompolicydata')
                if os.path.exists(default_file_path):
                    uploaded_file = open(default_file_path, 'rb')
                else:
                    st.error("Please upload a CSV file in the Configuration tab before running the simulation.")
                    st.session_state.simulation_run = False
                    return

            # Clear the warning flag for each new simulation
            if hasattr(simulate_lapse_month, '_warning_shown'):
                delattr(simulate_lapse_month, '_warning_shown')

            try:
                with st.spinner('Processing historical data and running projection...'):
                    
                    # 1. Historical Data Modeling
                    historical_df = load_and_process_data(uploaded_file, custom_survival_curves)
                    
                    if historical_df.empty:
                        st.warning("Simulation aborted because no valid historical policies were processed.")
                        st.session_state.simulation_run = False
                        return

                    # 2. Future Policy Generation (using variable volume)
                    future_df = generate_future_policies(
                        historical_df, 
                        initial_policies, 
                        weekly_change, 
                        ceiling,
                        volume_randomness,
                        custom_survival_curves
                    )
                    
                    # 3. Combine Data
                    common_cols = [
                        'PolicyNumber', 'AnnualPremium', 'UWClass', 'IssueDate', 'ApplicationDate', 
                        'Advance', 'TermDate', 'EstLapseDate', 'ActualMonthsSurvived',
                        'EarnedCommission', 'UnearnedCommission', 'Status'
                    ] + [f'Paidm{i}' for i in range(1, 13)]
                    
                    # Ensure all common columns exist before concatenation
                    historical_df = historical_df.reindex(columns=common_cols)
                    future_df = future_df.reindex(columns=common_cols)

                    all_policies_df = pd.concat([
                        historical_df,
                        future_df
                    ], ignore_index=True)
                    
                    # 4. Cash Flow Projection
                    projection_df = project_cash_flow(all_policies_df)

                    # 5. Survival Curve Analysis
                    survival_df_melted = generate_survival_curves(all_policies_df)
                    
                    # 6. Volume by Underwriting Class
                    volume_by_class_df = generate_volume_by_class(all_policies_df)
                
                # --- Results Display ---
                st.success("✅ Simulation completed successfully!")
                
                # Summary metrics
                st.subheader("📊 Key Metrics Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_advances = projection_df['AccumulatedAdvances'].iloc[-1]
                    st.metric("Total Advances", f"${total_advances:,.0f}")
                
                with col2:
                    total_chargebacks = projection_df['AccumulatedChargebacks'].iloc[-1]
                    st.metric("Total Chargebacks", f"${total_chargebacks:,.0f}")
                
                with col3:
                    final_unearned = projection_df['WeeklyUnearnedCommission'].iloc[-1]
                    st.metric("Final Unearned", f"${final_unearned:,.0f}")
                
                with col4:
                    net_cash = projection_df['AccumulatedNetCashFlow'].iloc[-1]
                    st.metric("Net Cash Flow", f"${net_cash:,.0f}")
                
                st.markdown("---")

                # --- Chart 1: Comprehensive Accumulated Metrics ---
                st.subheader("1. Accumulated Cash Flow and Commission Metrics")
                
                # Calculate accumulated cash (advances - chargebacks)
                projection_df['AccumulatedCash'] = projection_df['AccumulatedAdvances'] - projection_df['AccumulatedChargebacks']
                
                # Prepare data for plotting
                comprehensive_metrics = projection_df[[
                    'WeekEndDate', 
                    'AccumulatedAdvances',
                    'AccumulatedChargebacks',
                    'AccumulatedEarnedCommission',
                    'WeeklyUnearnedCommission',
                    'AccumulatedCash'
                ]].copy()
                
                # Rename for clearer legend
                comprehensive_metrics = comprehensive_metrics.rename(columns={
                    'AccumulatedAdvances': 'Accumulated Advances',
                    'AccumulatedChargebacks': 'Accumulated Chargebacks',
                    'AccumulatedEarnedCommission': 'Accumulated Earned Commission',
                    'WeeklyUnearnedCommission': 'Accumulated Unearned Commission',
                    'AccumulatedCash': 'Accumulated Cash (Net)'
                })
                
                # Create Plotly figure
                fig_comprehensive = px.line(
                    comprehensive_metrics,
                    x='WeekEndDate',
                    y=['Accumulated Advances', 'Accumulated Chargebacks', 'Accumulated Earned Commission', 
                       'Accumulated Unearned Commission', 'Accumulated Cash (Net)'],
                    labels={'value': 'Amount ($)', 'WeekEndDate': 'Week Ending', 'variable': 'Metric'}
                )
                
                # Add vertical line at historical cutoff
                fig_comprehensive.add_vline(
                    x=HISTORICAL_CUTOFF_DATE.timestamp() * 1000,
                    line_width=2,
                    line_dash="solid",
                    line_color="white",
                    annotation_text="Historical | Simulated",
                    annotation_position="top"
                )
                
                fig_comprehensive.update_layout(
                    hovermode='x unified',
                    legend_title_text='Metric'
                )
                
                st.plotly_chart(fig_comprehensive, use_container_width=True)
                st.info("📊 This chart shows all key accumulated metrics: **Advances** (total received), **Chargebacks** (total returned), **Earned Commission** (secured), **Unearned Commission** (at-risk), and **Accumulated Cash** (net advances minus chargebacks). The gap between Accumulated Advances and Accumulated Earned Commission represents your unearned liability.")
                
                # --- Chart 2: Cash Flow Time Series ---
                st.subheader("2. Weekly Cash Flow: Advances, Chargebacks, and Net Flow")
                
                cash_flow_metrics = projection_df[[
                    'WeekEndDate', 
                    'WeeklyAdvances', 
                    'WeeklyChargebacks', 
                    'WeeklyNetCashFlow'
                ]].copy()
                
                # Create Plotly bar chart
                fig_cash = px.bar(
                    cash_flow_metrics,
                    x='WeekEndDate',
                    y=['WeeklyAdvances', 'WeeklyChargebacks', 'WeeklyNetCashFlow'],
                    labels={'value': 'Amount ($)', 'WeekEndDate': 'Week Ending', 'variable': 'Metric'},
                    barmode='group'
                )
                
                # Add vertical line at historical cutoff
                fig_cash.add_vline(
                    x=HISTORICAL_CUTOFF_DATE.timestamp() * 1000,
                    line_width=2,
                    line_dash="solid",
                    line_color="white",
                    annotation_text="Historical | Simulated",
                    annotation_position="top"
                )
                
                fig_cash.update_layout(
                    hovermode='x unified',
                    legend_title_text='Metric'
                )
                
                st.plotly_chart(fig_cash, use_container_width=True)
                
                # --- Chart 3: Survival Curves ---
                st.subheader("3. 12-Month Policy Survival by Weekly Cohort")
                
                if not survival_df_melted.empty:
                    fig = px.line(
                        survival_df_melted, 
                        x="MonthNumber", 
                        y="SurvivalRate", 
                        color="IssueWeekStr", 
                        line_group="IssueWeekStr",
                        hover_name="IssueWeekStr",
                        title="Survival Rate over 12 Months for Weekly Cohorts (Based on Average PaidmX)",
                        labels={'MonthNumber': 'Policy Month (M1 to M12)', 'SurvivalRate': 'Survival Rate (Cumulative)', 'IssueWeekStr': 'Issue Week'},
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                    fig.update_layout(showlegend=False)
                    fig.update_yaxes(range=[0, 1])
                    st.plotly_chart(fig, use_container_width=True)
                    st.info("📊 This chart shows actual survival rates based on Bernoulli trial simulations. Each policy's PaidmX indicators were determined through month-by-month conditional lapse probability trials, ensuring statistically accurate representation of the survival curves.")
                else:
                    st.warning("No policies generated to display survival curves.")
                
                # --- Chart 4: Volume by Underwriting Class ---
                st.subheader("4. Weekly Policy Volume by Underwriting Class")
                
                if not volume_by_class_df.empty:
                    fig_volume = px.bar(
                        volume_by_class_df,
                        x='ApplicationWeek',
                        y=['Preferred', 'Standard', 'Graded', 'GI'],
                        title="Policy Volume by Underwriting Class (Stacked)",
                        labels={'value': 'Number of Policies', 'ApplicationWeek': 'Application Week', 'variable': 'UW Class'},
                        color_discrete_map={
                            'Preferred': '#1f77b4',
                            'Standard': '#ff7f0e', 
                            'Graded': '#2ca02c',
                            'GI': '#d62728'
                        }
                    )
                    
                    # Add vertical line at historical cutoff
                    fig_volume.add_vline(
                        x=HISTORICAL_CUTOFF_DATE.timestamp() * 1000,
                        line_width=2,
                        line_dash="solid",
                        line_color="white",
                        annotation_text="Historical | Simulated",
                        annotation_position="top"
                    )
                    
                    fig_volume.update_layout(
                        barmode='stack',
                        xaxis_title="Application Week",
                        yaxis_title="Number of Policies",
                        legend_title="Underwriting Class",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_volume, use_container_width=True)
                    st.info("📊 This stacked bar chart shows the weekly volume of policies by application date (when advances were received), broken down by underwriting class. Historical data shows actual application dates (app_recvd_date), while projected data shows simulated application dates. This aligns with the cash flow timing.")
                else:
                    st.warning("No volume data available to display.")
                
                # --- Chart 5: Average Weekly Cash Flows (Moving Average) ---
                st.subheader("5. Average Weekly Cash Flows (4-Week Moving Average)")
                
                # Calculate moving averages
                window_size = 4
                projection_df_ma = projection_df.copy()
                projection_df_ma['AvgAdvances'] = projection_df_ma['WeeklyAdvances'].rolling(window=window_size, min_periods=1).mean()
                projection_df_ma['AvgChargebacks'] = projection_df_ma['WeeklyChargebacks'].rolling(window=window_size, min_periods=1).mean()
                projection_df_ma['AvgNetCashFlow'] = projection_df_ma['WeeklyNetCashFlow'].rolling(window=window_size, min_periods=1).mean()
                
                avg_cash_flow_metrics = projection_df_ma[[
                    'WeekEndDate', 
                    'AvgAdvances', 
                    'AvgChargebacks', 
                    'AvgNetCashFlow'
                ]].copy()
                
                # Create Plotly figure
                fig_avg = px.line(
                    avg_cash_flow_metrics,
                    x='WeekEndDate',
                    y=['AvgAdvances', 'AvgChargebacks', 'AvgNetCashFlow'],
                    labels={'value': 'Amount ($)', 'WeekEndDate': 'Week Ending', 'variable': 'Metric'}
                )
                
                # Add vertical line at historical cutoff
                fig_avg.add_vline(
                    x=HISTORICAL_CUTOFF_DATE.timestamp() * 1000,
                    line_width=2,
                    line_dash="solid",
                    line_color="white",
                    annotation_text="Historical | Simulated",
                    annotation_position="top"
                )
                
                fig_avg.update_layout(
                    hovermode='x unified',
                    legend_title_text='Metric'
                )
                
                st.plotly_chart(fig_avg, use_container_width=True)
                st.info("📊 This chart shows 4-week moving averages to smooth out weekly volatility and reveal underlying trends in advances, chargebacks, and net cash flow over time.")

                st.markdown("---")

                # --- Download Final Data ---
                st.subheader("💾 Download Simulated Book of Business")
                st.markdown("Download the full policy-level data, including estimated lapse dates and **PaidmX** indicators, for detailed analysis.")

                csv_data = all_policies_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Simulated Book of Business CSV",
                    data=csv_data,
                    file_name="simulated_book_of_business.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # Download projection data
                projection_csv = projection_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Weekly Projection CSV",
                    data=projection_csv,
                    file_name="weekly_cash_flow_projection.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"An unexpected error occurred during processing.")
                st.exception(e)
                st.session_state.simulation_run = False

if __name__ == "__main__":
    main()
