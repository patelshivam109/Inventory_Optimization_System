import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# __file__ is dashboard/app.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # Points to 'dashboard'
ROOT_DIR = os.path.dirname(CURRENT_DIR)                  # Points to project root

# Define precise absolute paths
DEMAND_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "demand_forecast.csv")
INVENTORY_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "final_inventory_optimization.csv")
PRICING_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "pricing_optimized.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "demand_forecasting2_model.pkl")
FIGURES_DIR = os.path.join(ROOT_DIR, "reports", "figures")


# GENERAL CONFIGURATION

st.set_page_config(
    page_title="AI Powered Retail Inventory Optimization System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global Header
st.title("📊 AI Powered Retail Inventory Optimization System")
st.markdown("---")


# CACHED DATA & MODEL LOADING FUNCTIONS

@st.cache_data
def load_demand_data():
    try:
        if not os.path.exists(DEMAND_DATA_PATH):
            st.error("❌ Demand dataset not found. Please check your data directory.")
            return None
        df = pd.read_csv(DEMAND_DATA_PATH)
        for col in ['Date', 'date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        st.sidebar.success(f" Demand data loaded: {len(df)} rows")
        return df
    except Exception as e:
        st.error(f"Error loading demand data: {str(e)}")
        return None

@st.cache_data
def load_inventory_data():
    try:
        if not os.path.exists(INVENTORY_DATA_PATH):
            st.error("❌ Inventory dataset not found. Please check your data directory.")
            return None
        df = pd.read_csv(INVENTORY_DATA_PATH)
        st.sidebar.success(f" Inventory data loaded: {len(df)} rows")
        return df
    except Exception as e:
        st.error(f"Error loading inventory data: {str(e)}")
        return None

@st.cache_data
def load_pricing_data():
    try:
        if not os.path.exists(PRICING_DATA_PATH):
            st.error("❌ Pricing dataset not found. Please check your data directory.")
            return None
        df = pd.read_csv(PRICING_DATA_PATH)
        st.sidebar.success(f" Pricing data loaded: {len(df)} rows")
        return df
    except Exception as e:
        st.error(f"Error loading pricing data: {str(e)}")
        return None

@st.cache_resource
def load_ml_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.warning("⚠️ ML model not found. Please verify the model asset is present.")
            return None
        # Using joblib as requested for the production model
        model = joblib.load(MODEL_PATH)
        st.sidebar.success(f" ML Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Error loading ML model: {str(e)}")
        return None

# Helper function to render pre-saved figures safely
def display_saved_figure(subfolder, filename, caption):
    full_path = os.path.join(FIGURES_DIR, subfolder, filename)
    if os.path.exists(full_path):
        st.image(full_path, caption=caption, use_container_width=True)
    else:
        st.caption(f"ℹ️ Pre-saved chart '{filename}' not found at path.")

# Helper function to match column names case-insensitively
def get_col(df, target_names, default=""):
    for name in target_names:
        if name in df.columns:
            return name
        for c in df.columns:
            if c.lower().strip() == name.lower().strip():
                return c
    return default


# MODEL PREDICTION FUNCTION

def prepare_features_for_model(df):
    """
    Prepare features for demand forecasting model
    Expected input columns: Product ID, Store ID, Price, Promotions, 
    Seasonality Factors, External Factors, Demand Trend, Customer Segments
    """
    try:
        # Get column names (case-insensitive)
        prod_col = get_col(df, ['Product ID', 'product_id'])
        store_col = get_col(df, ['Store ID', 'store_id'])
        price_col = get_col(df, ['Price', 'price'])
        promo_col = get_col(df, ['Promotions', 'promotions'])
        season_col = get_col(df, ['Seasonality Factors', 'seasonality_factors'])
        external_col = get_col(df, ['External Factors', 'external_factors'])
        trend_col = get_col(df, ['Demand Trend', 'demand_trend'])
        segment_col = get_col(df, ['Customer Segments', 'customer_segments'])
        
        # Create feature dataframe with only required columns
        feature_cols = [prod_col, store_col, price_col, promo_col, season_col, external_col, trend_col, segment_col]
        feature_cols = [c for c in feature_cols if c]  # Remove empty strings
        
        if not feature_cols:
            return None
        
        X = df[feature_cols].copy()
        
        # Handle missing values
        X = X.fillna(X.mode().iloc[0])
        
        # One-hot encode categorical variables (same as training)
        categorical_cols = [promo_col, season_col, external_col, trend_col, segment_col]
        categorical_cols = [c for c in categorical_cols if c]
        
        if categorical_cols:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        return X
    except Exception as e:
        st.error(f"Error preparing features: {str(e)}")
        return None

def make_model_predictions(df, model):
    """Make demand predictions using the loaded model"""
    try:
        if model is None:
            return None
        
        X = prepare_features_for_model(df)
        if X is None or X.empty:
            return None

        # Align one-hot encoded features with model feature names
        def align_features(X, model):
            """Ensure X contains the same columns (and order) the model was trained on.

            - If the model exposes `feature_names_in_` (scikit-learn), use that as the required set.
            - Add missing columns with zeros and drop any extras.
            """
            try:
                if hasattr(model, 'feature_names_in_'):
                    required = list(model.feature_names_in_)
                else:
                    # If we don't know expected names, return X as-is
                    return X

                # Add missing cols with zeros
                for col in required:
                    if col not in X.columns:
                        X[col] = 0

                # Keep only required cols and in the correct order
                X = X.loc[:, required]
                return X
            except Exception as e:
                st.warning(f"Feature alignment warning: {str(e)}")
                return X

        X_aligned = align_features(X, model)
        try:
            predictions = model.predict(X_aligned)
            return predictions
        except Exception as e:
            # Likely feature mismatch — surface a helpful message
            st.error("Could not generate prediction. The model expects the same one-hot encoded feature columns used during training.")
            st.caption("Tip: Ensure categorical levels (e.g. Customer Segments_Premium) are present or will be added automatically as zeros.")
            st.warning(str(e))
            return None
    except Exception as e:
        st.warning(f"Model prediction error: {str(e)}")
        return None

# Load datasets
df_demand = load_demand_data()
df_inventory = load_inventory_data()
df_pricing = load_pricing_data()
model_lr = load_ml_model()


# SIDEBAR NAVIGATION

st.sidebar.title("Navigation Menu")

# Add a debug info expander in sidebar
with st.sidebar.expander("🔧 System Diagnostics", expanded=False):
    st.markdown("**Data Load Status:**")
    st.caption(f"✓ Demand dataset loaded: {df_demand is not None}")
    st.caption(f"✓ Inventory dataset loaded: {df_inventory is not None}")
    st.caption(f"✓ Pricing dataset loaded: {df_pricing is not None}")
    st.caption(f"✓ Machine learning model loaded: {model_lr is not None}")
    if df_demand is not None:
        st.caption(f"• Demand rows: {df_demand.shape[0]:,}")
    if df_inventory is not None:
        st.caption(f"• Inventory rows: {df_inventory.shape[0]:,}")
    if df_pricing is not None:
        st.caption(f"• Pricing rows: {df_pricing.shape[0]:,}")

page = st.sidebar.radio(
    "Select Dashboard Page:",
    [
        "Executive Overview",
        "Demand Forecasting",
        "Demand Prediction",
        "Inventory Optimization",
        "Pricing Optimization",
        "Data Explorer"
    ]
)


# PAGE 1: EXECUTIVE OVERVIEW

if "Executive Overview" in page or page == "🏠 Executive Overview":
    st.header(" Executive Overview")
    
    if df_demand is None or df_inventory is None or df_pricing is None:
        st.error("🚨 Missing processed data. Please run your preprocessing pipelines first.")
    else:
        prod_col_inv = get_col(df_inventory, ['Product ID', 'product_id'])
        store_col_inv = get_col(df_inventory, ['Store ID', 'store_id'])
        status_col = get_col(df_inventory, ['Inventory Status', 'inventory_status'])
        stock_level_col = get_col(df_inventory, ['Stock Levels', 'stock_levels'])
        safety_stock_col = get_col(df_inventory, ['Safety Stock', 'safety_stock'])
        price_rec_col = get_col(df_pricing, ['Pricing Recommendation', 'pricing_recommendation'])
        forecast_col = get_col(df_demand, ['Forecasted Demand', 'forecasted_demand'])
        
        # Compute Core KPIs
        total_products = len(set(df_inventory[prod_col_inv].dropna().unique()).union(set(df_pricing[get_col(df_pricing, ['Product ID', 'product_id'])].dropna().unique())))
        total_stores = df_inventory[store_col_inv].nunique() if store_col_inv else 0
        total_forecasted_demand = df_demand[forecast_col].sum() if forecast_col else 0
        
        low_stock_count = df_inventory[df_inventory[status_col].str.lower().str.contains('reorder', na=False)].shape[0] if status_col else 0
        stockout_count = df_inventory[df_inventory[status_col].str.lower().str.contains('stockout', na=False)].shape[0] if status_col else 0
        pricing_recommendations = df_pricing.shape[0]
        
        total_safety_stock = df_inventory[safety_stock_col].sum() if safety_stock_col else 0

        # KPI Cards with Better Formatting
        st.markdown("### 📊 Key Performance Indicators")
        kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
        
        with kpi1:
            st.metric(" Total Products", f"{total_products:,}")
        with kpi2:
            st.metric(" Total Stores", f"{total_stores:,}")
        with kpi3:
            st.metric(" Forecasted Demand", f"{total_forecasted_demand:,.0f}")
        with kpi4:
            st.metric(" Low Stock Items", f"{low_stock_count:,}", delta=f"{low_stock_count}", delta_color="inverse")
        with kpi5:
            st.metric(" Stockout Risk", f"{stockout_count:,}", delta=f"{stockout_count}", delta_color="inverse")
        with kpi6:
            st.metric(" Price Recommendations", f"{pricing_recommendations:,}")

        st.markdown("---")
        st.subheader("Executive Distributions")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            if status_col:
                status_counts = df_inventory[status_col].value_counts().reset_index()
                status_counts.columns = ['Status', 'Count']
                fig_inv = px.pie(status_counts, values='Count', names='Status', hole=0.4, title="Current Inventory Status Breakdown", color_discrete_sequence=px.colors.sequential.RdBu)
                st.plotly_chart(fig_inv, use_container_width=True)

        with chart_col2:
            if price_rec_col:
                price_counts = df_pricing[price_rec_col].value_counts().reset_index()
                price_counts.columns = ['Recommendation', 'Count']
                fig_price = px.bar(price_counts, x='Recommendation', y='Count', title="Recommended Pricing Strategies", color='Recommendation', color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_price, use_container_width=True)

        # Additional Analysis
        st.markdown("---")
        st.subheader("Demand & Inventory Trends")
        anal_col1, anal_col2 = st.columns(2)
        
        with anal_col1:
            if stock_level_col:
                # Stock level by status
                stock_by_status = df_inventory.groupby(status_col)[stock_level_col].mean().reset_index()
                stock_by_status.columns = ['Status', 'Avg Stock Level']
                fig_stock_status = px.bar(stock_by_status, x='Status', y='Avg Stock Level', 
                                          title="Average Stock Level by Inventory Status",
                                          color='Avg Stock Level', color_continuous_scale='Blues')
                st.plotly_chart(fig_stock_status, use_container_width=True)
        
        with anal_col2:
            if forecast_col and forecast_col in df_demand.columns:
                # Top products by forecast demand
                top_prod_forecast = df_demand.groupby(get_col(df_demand, ['Product ID', 'product_id']))[forecast_col].sum().reset_index().sort_values(by=forecast_col, ascending=False).head(10)
                top_prod_forecast.columns = ['Product', 'Forecasted Demand']
                fig_top_forecast = px.bar(top_prod_forecast, x='Product', y='Forecasted Demand',
                                         title="Top 10 Products by Forecasted Demand",
                                         color='Forecasted Demand', color_continuous_scale='Viridis')
                st.plotly_chart(fig_top_forecast, use_container_width=True)


# PAGE 2: DEMAND FORECASTING

elif page == "Demand Forecasting":
    st.header("Demand Forecasting & ML Performance")
    
    if df_demand is None:
        st.error("Could not load demand dataset. Please check your data source.")
    else:
        prod_col = get_col(df_demand, ['Product ID', 'product_id'])
        store_col = get_col(df_demand, ['Store ID', 'store_id'])
        date_col = get_col(df_demand, ['Date', 'date'])
        actual_col = get_col(df_demand, ['Sales Quantity', 'Actual Sales Quantity', 'sales_quantity'])
        forecast_col = get_col(df_demand, ['Forecasted Demand', 'forecasted_demand'])

        # Filters simplified: show full demand dataset by default
        filtered_demand = df_demand.copy()

        # Model Performance Metrics
        st.subheader("Model Performance Evaluation Metrics")
        
        # Make predictions using the loaded model
        if model_lr is not None:
            st.info(" Using **demand_forecasting2_model.pkl** for live predictions")
            model_predictions = make_model_predictions(filtered_demand, model_lr)
            
            if model_predictions is not None:
                # Add model predictions to dataframe
                comparison_df = filtered_demand.copy()
                comparison_df['Model Predictions'] = model_predictions
                
                eval_df = comparison_df.dropna(subset=[actual_col, 'Model Predictions'])
                
                if not eval_df.empty:
                    y_true = eval_df[actual_col]
                    y_pred_model = eval_df['Model Predictions']
                    y_pred_csv = eval_df[forecast_col] if forecast_col else None
                    
                    col_mae, col_rmse, col_r2 = st.columns(3)
                    col_mae.metric("Model MAE", f"{mean_absolute_error(y_true, y_pred_model):.2f}")
                    col_rmse.metric("Model RMSE", f"{np.sqrt(mean_squared_error(y_true, y_pred_model)):.2f}")
                    col_r2.metric("Model R² Score", f"{r2_score(y_true, y_pred_model):.4f}")
                    
                    # Display actual vs predicted comparison details
                    st.markdown("---")
                    st.subheader(" Actual vs Predicted Results")
                    detail_display = eval_df[[actual_col, 'Model Predictions']].copy()
                    detail_display = detail_display.rename(columns={
                        actual_col: 'Actual',
                        'Model Predictions': 'Predicted'
                    })
                    if forecast_col is not None and forecast_col in eval_df.columns:
                        detail_display['Forecasted (CSV)'] = eval_df[forecast_col]
                    detail_display['Error'] = (detail_display['Actual'] - detail_display['Predicted']).abs()
                    st.dataframe(
                        detail_display.head(20).style.format({
                            'Actual': '{:.0f}',
                            'Predicted': '{:.2f}',
                            'Forecasted (CSV)': '{:.2f}',
                            'Error': '{:.2f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Comparison with CSV forecasts if available
                    if y_pred_csv is not None:
                        st.markdown("---")
                        st.subheader("📊 Prediction Comparison: Model vs CSV Forecasts")
                        
                        csv_mae = mean_absolute_error(y_true, y_pred_csv)
                        model_mae = mean_absolute_error(y_true, y_pred_model)
                        
                        comp_col1, comp_col2, comp_col3 = st.columns(3)
                        comp_col1.metric("CSV Forecasts MAE", f"{csv_mae:.2f}")
                        comp_col2.metric("Model Predictions MAE", f"{model_mae:.2f}")
                        comp_col3.metric("Improvement", f"{((csv_mae - model_mae) / csv_mae * 100):.1f}%", 
                                        delta="Better" if model_mae < csv_mae else "Worse")
                else:
                    st.warning("Insufficient data for model predictions.")
            else:
                st.warning("Could not generate model predictions - feature mismatch or data issue.")
        else:
            # Fallback to CSV forecasts if model not loaded
            eval_df = filtered_demand.dropna(subset=[actual_col, forecast_col]) if (actual_col and forecast_col) else pd.DataFrame()
            
            if not eval_df.empty:
                y_true = eval_df[actual_col]
                y_pred = eval_df[forecast_col]
                m1, m2, m3 = st.columns(3)
                m1.metric("Mean Absolute Error (MAE)", f"{mean_absolute_error(y_true, y_pred):.2f}")
                m2.metric("Root Mean Squared Error (RMSE)", f"{np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
                m3.metric("R² Score", f"{r2_score(y_true, y_pred):.4f}")
            else:
                st.warning("Insufficient actual/forecast overlapping rows to generate performance profiles.")

        # Interactive Timeline Chart
        st.subheader("Sales Quantity vs Forecasted Demand Timeline")
        if date_col and actual_col:
            timeline_df = filtered_demand.copy()
            
            # Add model predictions to timeline
            if model_lr is not None:
                timeline_df['Model Predictions'] = make_model_predictions(filtered_demand, model_lr)
            
            # Group by date and sum
            timeline_grouped = timeline_df.groupby(date_col).agg({
                actual_col: 'sum',
                forecast_col: 'sum' if forecast_col else 'first'
            }).reset_index()
            
            if model_lr is not None and 'Model Predictions' in timeline_df.columns:
                timeline_grouped['Model Predictions'] = timeline_df.groupby(date_col)['Model Predictions'].sum().values
                y_cols = [actual_col, forecast_col, 'Model Predictions']
            else:
                y_cols = [actual_col, forecast_col]
            
            timeline_grouped = timeline_grouped.sort_values(by=date_col)
            fig_timeline = px.line(timeline_grouped, x=date_col, y=y_cols, 
                                   labels={"value": "Quantity", "variable": "Data Stream"},
                                   title="Actual vs Forecasted vs Model Predictions")
            st.plotly_chart(fig_timeline, use_container_width=True)

        # Top 10 Forecasted Demand Products
        st.subheader("Top 10 Products by Forecasted Demand")
        if prod_col and forecast_col:
            top_10 = filtered_demand.groupby(prod_col)[forecast_col].sum().reset_index().sort_values(by=forecast_col, ascending=False).head(10)
            fig_top10 = px.bar(top_10, x=prod_col, y=forecast_col, color=forecast_col, color_continuous_scale='Viridis')
            st.plotly_chart(fig_top10, use_container_width=True)

        # Model Predictions Detail View
        if model_lr is not None:
            st.markdown("---")
            st.subheader("🔍 Model Predictions Detail View")
            
            detail_df = filtered_demand.copy()
            detail_df['Model Prediction'] = make_model_predictions(filtered_demand, model_lr)
            
            if actual_col and detail_df['Model Prediction'].notna().any():
                # Calculate error metrics for each row
                detail_df['Actual Sales'] = detail_df[actual_col]
                detail_df['CSV Forecast'] = detail_df[forecast_col] if forecast_col else 0
                detail_df['Error (Actual vs Model)'] = abs(detail_df['Actual Sales'] - detail_df['Model Prediction'])
                
                # Display key columns
                display_detail_cols = []
                if prod_col:
                    display_detail_cols.append(prod_col)
                if store_col:
                    display_detail_cols.append(store_col)
                display_detail_cols.extend(['Actual Sales', 'Model Prediction', 'CSV Forecast', 'Error (Actual vs Model)'])
                
                display_detail_cols = [c for c in display_detail_cols if c in detail_df.columns]
                
                st.dataframe(
                    detail_df[display_detail_cols].head(20).style.format({
                        'Actual Sales': '{:.0f}',
                        'Model Prediction': '{:.2f}',
                        'CSV Forecast': '{:.2f}',
                        'Error (Actual vs Model)': '{:.2f}'
                    }),
                    use_container_width=True
                )

        # Pre-saved Figures Section
        st.markdown("---")
        st.subheader("📁 Pre-Saved Statistical Insights")
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            display_saved_figure("demand", "sales_quantity_distribution.png", "Sales Quantity Volumetric Distribution")
            display_saved_figure("demand", "Top 10 Fast Moving Products.png", "Top 10 Fast Moving Products Profile")
        with img_col2:
            display_saved_figure("demand", "demand_trend_distribution.png", "Long-Term Demand Trends Distribution")
            display_saved_figure("demand", "Top 10 Slow Moving Products.png", "Top 10 Slow Moving Products Profile")


# PAGE 3: DEMAND PREDICTION
elif page == "Demand Prediction":
    st.header(" Demand Prediction")

    if df_demand is None:
        st.error("Could not load demand dataset. Please check your data source.")
    elif model_lr is None:
        st.error("Could not load the forecasting model. Please verify the model file.")
    else:
        prod_col = get_col(df_demand, ['Product ID', 'product_id'])
        store_col = get_col(df_demand, ['Store ID', 'store_id'])
        date_col = get_col(df_demand, ['Date', 'date'])
        actual_col = get_col(df_demand, ['Sales Quantity', 'Actual Sales Quantity', 'sales_quantity'])
        price_col = get_col(df_demand, ['Price', 'price'])
        promo_col = get_col(df_demand, ['Promotions', 'promotions'])
        season_col = get_col(df_demand, ['Seasonality Factors', 'seasonality_factors'])
        external_col = get_col(df_demand, ['External Factors', 'external_factors'])
        trend_col = get_col(df_demand, ['Demand Trend', 'demand_trend'])
        segment_col = get_col(df_demand, ['Customer Segments', 'customer_segments'])

        if not prod_col or not store_col or not price_col:
            st.error("Required demand input columns are missing. Please verify the dataset schema.")
        else:
            st.markdown("Use this page to generate a clean, focused demand forecast for a specific product and store.")

            product_options = sorted(df_demand[prod_col].dropna().astype(str).unique())
            selected_product = st.selectbox("Product ID", ["All"] + product_options)

            store_subset = df_demand if selected_product == "All" else df_demand[df_demand[prod_col].astype(str) == selected_product]
            store_options = sorted(store_subset[store_col].dropna().astype(str).unique())
            selected_store = st.selectbox("Store ID", ["All"] + store_options)

            if selected_product == "All" or selected_store == "All":
                st.info("Select both Product ID and Store ID to generate a forecast.")
            else:
                def compute_defaults(product, store):
                    defaults = {}
                    try:
                        subset = df_demand[(df_demand[prod_col].astype(str) == str(product)) & (df_demand[store_col].astype(str) == str(store))]
                        if subset.empty:
                            subset = df_demand[df_demand[prod_col].astype(str) == str(product)]
                        if subset.empty:
                            subset = df_demand

                        defaults['price'] = float(subset[price_col].median()) if price_col in subset.columns else 0.0
                        defaults['promotions'] = subset[promo_col].mode().iloc[0] if promo_col and promo_col in subset.columns and not subset[promo_col].dropna().empty else None
                        defaults['seasonality_factors'] = subset[season_col].mode().iloc[0] if season_col and season_col in subset.columns and not subset[season_col].dropna().empty else None
                        defaults['external_factors'] = subset[external_col].mode().iloc[0] if external_col and external_col in subset.columns and not subset[external_col].dropna().empty else None
                        defaults['demand_trend'] = subset[trend_col].mode().iloc[0] if trend_col and trend_col in subset.columns and not subset[trend_col].dropna().empty else None
                        defaults['customer_segments'] = subset[segment_col].mode().iloc[0] if segment_col and segment_col in subset.columns and not subset[segment_col].dropna().empty else None
                    except Exception:
                        defaults = {'price': 0.0, 'promotions': None, 'seasonality_factors': None, 'external_factors': None, 'demand_trend': None, 'customer_segments': None}
                    return defaults

                defaults = compute_defaults(selected_product, selected_store)
                price_input = st.number_input("Price", min_value=0.0, value=float(defaults.get('price', 0.0)), step=0.5)

                with st.expander("Advanced demand feature values"):
                    pred_promo = st.selectbox("Promotions", sorted(df_demand[promo_col].dropna().astype(str).unique()) if promo_col and promo_col in df_demand.columns else [defaults.get('promotions')])
                    pred_season = st.selectbox("Seasonality", sorted(df_demand[season_col].dropna().astype(str).unique()) if season_col and season_col in df_demand.columns else [defaults.get('seasonality_factors')])
                    pred_external = st.selectbox("External Factors", sorted(df_demand[external_col].dropna().astype(str).unique()) if external_col and external_col in df_demand.columns else [defaults.get('external_factors')])
                    pred_trend = st.selectbox("Demand Trend", sorted(df_demand[trend_col].dropna().astype(str).unique()) if trend_col and trend_col in df_demand.columns else [defaults.get('demand_trend')])
                    pred_segment = st.selectbox("Customer Segment", sorted(df_demand[segment_col].dropna().astype(str).unique()) if segment_col and segment_col in df_demand.columns else [defaults.get('customer_segments')])

                if st.button("Generate Prediction"):
                    pred_input = pd.DataFrame({
                        prod_col: [selected_product],
                        store_col: [selected_store],
                        price_col: [price_input],
                        promo_col: [pred_promo],
                        season_col: [pred_season],
                        external_col: [pred_external],
                        trend_col: [pred_trend],
                        segment_col: [pred_segment],
                    })

                    prediction = make_model_predictions(pred_input, model_lr)
                    if prediction is not None and len(prediction) > 0:
                        st.success(f"Predicted Demand: **{prediction[0]:.2f} units**")
                        st.markdown("---")
                        st.subheader("Input Summary")
                        st.dataframe(pred_input.T, use_container_width=True)
                        actual_subset = df_demand[(df_demand[prod_col].astype(str) == selected_product) & (df_demand[store_col].astype(str) == selected_store)]
                        if not actual_subset.empty and actual_col in actual_subset.columns:
                            st.subheader("Recent Actual Demand")
                            actual_recent = actual_subset.sort_values(by=date_col if date_col in actual_subset.columns else actual_col, ascending=False).head(5)
                            actual_recent_display = actual_recent[[date_col, actual_col]].rename(columns={date_col: 'Date', actual_col: 'Actual Demand'}) if date_col in actual_subset.columns else actual_recent[[actual_col]].rename(columns={actual_col:'Actual Demand'})
                            st.dataframe(actual_recent_display, use_container_width=True)
                    else:
                        st.error("Prediction could not be generated. Please verify the input values and model compatibility.")


# PAGE 4: INVENTORY OPTIMIZATION

elif "Inventory Optimization" in page or page == "📦 Inventory Optimization":
    st.header("📦 Inventory Optimization & Safety Stock Strategy")
    
    if df_inventory is None:
        st.error("Could not load inventory dataset. Please check your data source.")
    else:
        p_id = get_col(df_inventory, ['Product ID', 'product_id'])
        s_id = get_col(df_inventory, ['Store ID', 'store_id'])
        s_level = get_col(df_inventory, ['Stock Levels', 'stock_levels'])
        f_dem = get_col(df_inventory, ['Forecasted Demand', 'forecasted_demand'])
        saf_stk = get_col(df_inventory, ['Safety Stock', 'safety_stock'])
        rec_re = get_col(df_inventory, ['Recommended Reorder Point', 'recommended_reorder_point'])
        inv_stat = get_col(df_inventory, ['Inventory Status', 'inventory_status'])

        # Data Cleaning
        df_inventory = df_inventory.fillna({s_level: 0, f_dem: 0, saf_stk: 0, rec_re: 0, inv_stat: 'Normal'})

        # Filter simplified: display full inventory by default
        filtered_inv = df_inventory.copy()

        # Highlighting Protocol for Critical Rows
        def highlight_critical(row):
            val = str(row[inv_stat]).lower() if inv_stat in row else ''
            if 'stockout' in val:
                return ['background-color: #f8d7da; color: #721c24; font-weight: bold'] * len(row)
            elif 'reorder' in val:
                return ['background-color: #fff3cd; color: #856404'] * len(row)
            return [''] * len(row)

        st.subheader("Inventory Replenishment Status Matrix")
        display_cols = [c for c in [p_id, s_id, s_level, f_dem, saf_stk, rec_re, inv_stat] if c]
        st.dataframe(filtered_inv[display_cols].style.apply(highlight_critical, axis=1), use_container_width=True)

        # Dynamic Interactive Visualizations
        st.markdown("### Real-Time Allocation Distributions")
        inv_col1, inv_col2 = st.columns(2)
        with inv_col1:
            if s_level:
                fig_stock = px.histogram(filtered_inv, x=s_level, nbins=30, title="Live Stock Level Density Distribution", color_discrete_sequence=['#1f77b4'])
                st.plotly_chart(fig_stock, use_container_width=True)
        with inv_col2:
            if s_id and s_level:
                store_allocations = filtered_inv.groupby(s_id)[s_level].sum().reset_index()
                fig_store = px.bar(store_allocations, x=s_id, y=s_level, title="Total Stock Volume by Store Location", color_continuous_scale='Cividis', color=s_level)
                st.plotly_chart(fig_store, use_container_width=True)

        # Pre-saved Operational Figures
        st.markdown("---")
        st.subheader("📁 Pre-Saved Operational Layouts")
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            display_saved_figure("inventory", "warehouse_utilization_distribution.png", "Warehouse Capacity Utilization Distribution")
            display_saved_figure("inventory", "stockout_frequency_distribution.png", "Historical Stockout Frequency Curve")
        with img_col2:
            display_saved_figure("inventory", "stock_level_distribution.png", "Static Stock Level Baseline Distribution")
            display_saved_figure("inventory", "reorder_point_distribution.png", "Calculated Reorder Point Range Analysis")


# PAGE 4: PRICING OPTIMIZATION

elif "Pricing Optimization" in page or page == " Pricing Optimization":
    st.header(" Pricing Intelligence & Elasticity Engine")
    
    if df_pricing is None:
        st.error("Pricing metrics profiles are unavailable. Please check your data source.")
    else:
        p_id = get_col(df_pricing, ['Product ID', 'product_id'])
        base_pr = get_col(df_pricing, ['Price', 'price'])
        comp_pr = get_col(df_pricing, ['Competitor Prices', 'competitor_prices'])
        elast_idx = get_col(df_pricing, ['Elasticity Index', 'elasticity_index'])
        rec_pr = get_col(df_pricing, ['Recommended Price', 'recommended_price'])
        pricing_rec = get_col(df_pricing, ['Pricing Recommendation', 'pricing_recommendation'])

        df_pricing = df_pricing.fillna({base_pr: 0, comp_pr: 0, elast_idx: 0, rec_pr: 0, pricing_rec: 'Maintain Value'})

        st.subheader("Strategic Recommendations Counting")
        if pricing_rec:
            rec_sums = df_pricing[pricing_rec].value_counts()
            cols_metrics = st.columns(len(rec_sums))
            for idx, (label, val) in enumerate(rec_sums.items()):
                cols_metrics[idx].metric(label=f"Action: {label}", value=int(val))

        st.subheader("Pricing Optimization Reference Matrix")
        display_cols_pr = [c for c in [p_id, base_pr, comp_pr, elast_idx, rec_pr, pricing_rec] if c]
        st.dataframe(df_pricing[display_cols_pr], use_container_width=True)

        # Pre-saved Pricing Figures
        st.markdown("---")
        st.subheader("📁 Pre-Saved Pricing Strategy Profiles")
        img_col1, img_col2, img_col3 = st.columns(3)
        with img_col1:
            display_saved_figure("pricing", "price_gap_distribution.png", "Price Gap Variance (Internal vs Competitor)")
        with img_col2:
            display_saved_figure("pricing", "elasticity_index_distribution.png", "Product Price Elasticity Indices")
        with img_col3:
            display_saved_figure("pricing", "price_vs_competitor_price.png", "Direct Price vs Competitor Pricing Alignment")


# PAGE 5: DATA EXPLORER

elif "Data Explorer" in page or page == "🗂️ Data Explorer":
    st.header("🗂️ Interactive Cross-Functional Data Explorer")
    
    dataset_selection = st.selectbox(
        "Choose Dataset to Explore:",
        ["Demand Data Profile", "Inventory Optimization Framework", "Pricing Metrics Index"]
    )

    if dataset_selection == "Demand Data Profile":
        active_df, file_ref_lbl = df_demand.copy(), DEMAND_DATA_PATH
    elif dataset_selection == "Inventory Optimization Framework":
        active_df, file_ref_lbl = df_inventory.copy(), INVENTORY_DATA_PATH
    else:
        active_df, file_ref_lbl = df_pricing.copy(), PRICING_DATA_PATH

    if active_df is None:
        st.error("Dataset could not be loaded. Please check the selected source.")
    else:
        st.markdown(f"📊 **Dataset:** {os.path.basename(file_ref_lbl)} | **Rows:** {active_df.shape[0]:,} | **Columns:** {active_df.shape[1]}")
        
        # Search and Filter Options
        st.subheader("🔍 Search & Filter Options")
        
        search_col1, search_col2 = st.columns(2)
        
        with search_col1:
            search_query = st.text_input("Search globally (all columns):", "")
        
        with search_col2:
            sort_column = st.selectbox("Sort by Column:", active_df.columns)
        
        explored_df = active_df.copy()
        
        # Text search filter
        if search_query:
            mask = explored_df.astype(str).apply(lambda row: row.str.contains(search_query, case=False).any(), axis=1)
            explored_df = explored_df[mask]
        
        # Sort
        if sort_column:
            try:
                explored_df = explored_df.sort_values(by=sort_column, ascending=True)
            except:
                pass
        
        # Display options
        display_col1, display_col2 = st.columns(2)
        with display_col1:
            rows_to_display = st.slider("Rows to Display", 10, min(len(explored_df), 500), 50)
        with display_col2:
            st.metric("Displayed Rows", f"{min(rows_to_display, len(explored_df))}/{len(explored_df)}")
        
        # Show data
        st.dataframe(explored_df.head(rows_to_display), use_container_width=True, height=400)
        
        # Download button
        csv_buffer = explored_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv_buffer,
            file_name=f"exported_{dataset_selection.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )


# GLOBAL FOOTER SYSTEM

st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("### 📊 Dashboard")
    st.caption("• 5 Intelligence Modules")
    st.caption("• Real-time Analytics")
    st.caption("• ML Predictions")

with footer_col2:
    st.markdown("### 🤖 AI Models")
    st.caption("• Demand Forecasting (LinearRegression)")
    st.caption("• Inventory Risk Detection")
    st.caption("• Pricing Optimization")

with footer_col3:
    st.markdown("###  Data")
    st.caption(f"• Demand: {df_demand.shape[0] if df_demand is not None else 0:,} records")
    st.caption(f"• Inventory: {df_inventory.shape[0] if df_inventory is not None else 0:,} records")
    st.caption(f"• Pricing: {df_pricing.shape[0] if df_pricing is not None else 0:,} records")

st.markdown("---")
st.caption("""
 **AI Powered Retail Inventory Optimization System** | Demand Forecasting • Inventory Optimization • Pricing Intelligence  
*Built with Streamlit, Plotly, and Scikit-learn for intelligent retail decision-making.*
""")