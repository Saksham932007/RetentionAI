"""
Data Overview page for RetentionAI Streamlit application.

This page provides comprehensive data exploration and visualization capabilities
including dataset statistics, distribution analysis, churn rate breakdowns,
feature correlations, and interactive filtering.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import sys
import logging

# Setup path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent))

try:
    from database import get_database_manager
    from config import PROCESSED_DATA_DIR
    from utils import load_json
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()

logger = logging.getLogger(__name__)


class DataOverviewPage:
    """Data exploration and visualization page for RetentionAI."""
    
    def __init__(self):
        """Initialize the data overview page."""
        self.db_manager = get_database_manager()
    
    @st.cache_data
    def load_dataset(_self, table_name: str = "processed_data") -> pd.DataFrame:
        """
        Load dataset from database with caching.
        
        Args:
            table_name: Database table name
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            query = f"SELECT * FROM {table_name}"
            df = _self.db_manager.execute_query(query)
            logger.info(f"Loaded {len(df)} records from {table_name}")
            return df
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return pd.DataFrame()
    
    def render_dataset_overview(self, df: pd.DataFrame) -> None:
        """Render dataset overview statistics."""
        
        st.markdown("### 📊 Dataset Overview")
        
        if df.empty:
            st.error("No data available. Please check database connection and ensure data has been ingested.")
            return
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        
        with col2:
            n_features = len([col for col in df.columns if col not in ['customerid', 'churn']])
            st.metric("Features", n_features)
        
        with col3:
            if 'churn' in df.columns:
                churn_rate = df['churn'].mean() if df['churn'].dtype in ['int64', 'float64'] else df['churn'].value_counts(normalize=True).get(1, 0)
                st.metric("Churn Rate", f"{churn_rate:.1%}", delta=f"{churn_rate:.1%}")
            else:
                st.metric("Churn Rate", "N/A")
        
        with col4:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Missing Data", f"{missing_pct:.1%}")
        
        # Data quality indicators
        st.markdown("#### Data Quality Indicators")
        
        quality_cols = st.columns(3)
        
        with quality_cols[0]:
            duplicate_count = df.duplicated().sum()
            duplicate_pct = (duplicate_count / len(df)) * 100
            if duplicate_pct < 1:
                st.success(f"✅ Low Duplicates: {duplicate_pct:.2f}%")
            else:
                st.warning(f"⚠️ Duplicates Found: {duplicate_pct:.2f}%")
        
        with quality_cols[1]:
            if missing_pct < 5:
                st.success(f"✅ Low Missing Data: {missing_pct:.2f}%")
            elif missing_pct < 15:
                st.warning(f"⚠️ Moderate Missing Data: {missing_pct:.2f}%")
            else:
                st.error(f"❌ High Missing Data: {missing_pct:.2f}%")
        
        with quality_cols[2]:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            outlier_count = 0
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
                outlier_count += len(outliers)
            
            outlier_pct = (outlier_count / len(df)) * 100 if len(df) > 0 else 0
            if outlier_pct < 5:
                st.success(f"✅ Low Outliers: {outlier_pct:.2f}%")
            else:
                st.warning(f"⚠️ Outliers Present: {outlier_pct:.2f}%")
    
    def render_churn_analysis(self, df: pd.DataFrame) -> None:
        """Render churn rate analysis."""
        
        st.markdown("### 🎯 Churn Analysis")
        
        if 'churn' not in df.columns:
            st.warning("Churn column not found in dataset")
            return
        
        # Overall churn distribution
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn pie chart
            churn_counts = df['churn'].value_counts()
            if df['churn'].dtype in ['int64', 'float64']:
                labels = ['Retained', 'Churned']
                values = [churn_counts.get(0, 0), churn_counts.get(1, 0)]
            else:
                labels = churn_counts.index.tolist()
                values = churn_counts.values.tolist()
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                marker_colors=['#2E8B57', '#DC143C']
            )])\
            fig_pie.update_layout(\n                title=\"Customer Churn Distribution\",\n                showlegend=True,\n                height=300\n            )\n            st.plotly_chart(fig_pie, use_container_width=True)\n        \n        with col2:\n            # Churn metrics\n            if df['churn'].dtype in ['int64', 'float64']:\n                churned_customers = df[df['churn'] == 1]\n                retained_customers = df[df['churn'] == 0]\n                \n                st.metric(\"Churned Customers\", f\"{len(churned_customers):,}\")\n                st.metric(\"Retained Customers\", f\"{len(retained_customers):,}\")\n                \n                churn_rate = len(churned_customers) / len(df) * 100\n                st.metric(\"Churn Rate\", f\"{churn_rate:.2f}%\")\n                \n                # Industry benchmark\n                st.markdown(\"**Industry Benchmark:** ~20-25%\")\n                if churn_rate < 20:\n                    st.success(\"Below industry average 👍\")\n                elif churn_rate < 30:\n                    st.warning(\"Around industry average\")\n                else:\n                    st.error(\"Above industry average ⚠️\")\n        \n        # Churn by categorical features\n        st.markdown(\"#### Churn Rate by Customer Segments\")\n        \n        categorical_cols = df.select_dtypes(include=['object']).columns\n        categorical_cols = [col for col in categorical_cols if col not in ['customerid', 'churn']]\n        \n        if len(categorical_cols) > 0:\n            selected_feature = st.selectbox(\n                \"Select feature for churn analysis:\",\n                categorical_cols,\n                key=\"churn_analysis_feature\"\n            )\n            \n            if selected_feature:\n                # Calculate churn rate by feature\n                if df['churn'].dtype in ['int64', 'float64']:\n                    churn_by_feature = df.groupby(selected_feature)['churn'].agg(['count', 'sum', 'mean']).round(3)\n                    churn_by_feature.columns = ['Total_Customers', 'Churned_Customers', 'Churn_Rate']\n                    churn_by_feature['Churn_Rate_Pct'] = churn_by_feature['Churn_Rate'] * 100\n                else:\n                    # Handle string churn values\n                    df_temp = df.copy()\n                    df_temp['churn_numeric'] = df_temp['churn'].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 1: 1, 0: 0})\n                    churn_by_feature = df_temp.groupby(selected_feature)['churn_numeric'].agg(['count', 'sum', 'mean']).round(3)\n                    churn_by_feature.columns = ['Total_Customers', 'Churned_Customers', 'Churn_Rate']\n                    churn_by_feature['Churn_Rate_Pct'] = churn_by_feature['Churn_Rate'] * 100\n                \n                # Bar chart\n                fig_bar = px.bar(\n                    x=churn_by_feature.index,\n                    y=churn_by_feature['Churn_Rate_Pct'],\n                    title=f\"Churn Rate by {selected_feature.title()}\",\n                    labels={'x': selected_feature.title(), 'y': 'Churn Rate (%)'}\n                )\n                fig_bar.update_layout(height=400)\n                st.plotly_chart(fig_bar, use_container_width=True)\n                \n                # Summary table\n                st.markdown(\"**Summary Table:**\")\n                st.dataframe(churn_by_feature, use_container_width=True)\n    \n    def render_feature_distributions(self, df: pd.DataFrame) -> None:\n        \"\"\"Render feature distribution analysis.\"\"\"\n        \n        st.markdown(\"### 📈 Feature Distributions\")\n        \n        # Numeric features\n        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()\n        numeric_cols = [col for col in numeric_cols if col not in ['customerid', 'churn']]\n        \n        if len(numeric_cols) > 0:\n            st.markdown(\"#### Numerical Features\")\n            \n            # Feature selection\n            selected_numeric = st.multiselect(\n                \"Select numerical features to visualize:\",\n                numeric_cols,\n                default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols,\n                key=\"numeric_features_selection\"\n            )\n            \n            if selected_numeric:\n                # Create subplots for histograms\n                n_cols = min(2, len(selected_numeric))\n                n_rows = (len(selected_numeric) + n_cols - 1) // n_cols\n                \n                fig = make_subplots(\n                    rows=n_rows, cols=n_cols,\n                    subplot_titles=selected_numeric\n                )\n                \n                for i, col in enumerate(selected_numeric):\n                    row = i // n_cols + 1\n                    col_pos = i % n_cols + 1\n                    \n                    # Add histogram\n                    fig.add_trace(\n                        go.Histogram(\n                            x=df[col],\n                            name=col,\n                            showlegend=False\n                        ),\n                        row=row, col=col_pos\n                    )\n                \n                fig.update_layout(\n                    height=300 * n_rows,\n                    title=\"Distribution of Numerical Features\"\n                )\n                st.plotly_chart(fig, use_container_width=True)\n        \n        # Categorical features\n        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()\n        categorical_cols = [col for col in categorical_cols if col not in ['customerid', 'churn']]\n        \n        if len(categorical_cols) > 0:\n            st.markdown(\"#### Categorical Features\")\n            \n            selected_categorical = st.selectbox(\n                \"Select categorical feature to visualize:\",\n                categorical_cols,\n                key=\"categorical_feature_selection\"\n            )\n            \n            if selected_categorical:\n                value_counts = df[selected_categorical].value_counts()\n                \n                # Bar chart\n                fig_cat = px.bar(\n                    x=value_counts.index,\n                    y=value_counts.values,\n                    title=f\"Distribution of {selected_categorical.title()}\",\n                    labels={'x': selected_categorical.title(), 'y': 'Count'}\n                )\n                fig_cat.update_layout(height=400)\n                st.plotly_chart(fig_cat, use_container_width=True)\n                \n                # Summary statistics\n                col1, col2 = st.columns(2)\n                with col1:\n                    st.metric(\"Unique Values\", len(value_counts))\n                with col2:\n                    st.metric(\"Most Common\", f\"{value_counts.index[0]} ({value_counts.iloc[0]:,})\")\n    \n    def render_correlation_analysis(self, df: pd.DataFrame) -> None:\n        \"\"\"Render correlation analysis.\"\"\"\n        \n        st.markdown(\"### 🔗 Feature Correlation Analysis\")\n        \n        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()\n        \n        if len(numeric_cols) < 2:\n            st.warning(\"Need at least 2 numerical columns for correlation analysis\")\n            return\n        \n        # Calculate correlation matrix\n        corr_matrix = df[numeric_cols].corr()\n        \n        # Correlation heatmap\n        fig_corr = px.imshow(\n            corr_matrix,\n            title=\"Feature Correlation Matrix\",\n            color_continuous_scale=\"RdBu_r\",\n            aspect=\"auto\"\n        )\n        fig_corr.update_layout(height=500)\n        st.plotly_chart(fig_corr, use_container_width=True)\n        \n        # Highest correlations\n        st.markdown(\"#### Highest Correlations\")\n        \n        # Get upper triangle of correlation matrix\n        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))\n        corr_pairs = corr_matrix.mask(mask).stack().reset_index()\n        corr_pairs.columns = ['Feature_1', 'Feature_2', 'Correlation']\n        corr_pairs = corr_pairs.sort_values('Correlation', key=abs, ascending=False)\n        \n        # Display top correlations\n        top_correlations = corr_pairs.head(10)\n        \n        col1, col2 = st.columns(2)\n        \n        with col1:\n            st.markdown(\"**Strongest Positive Correlations:**\")\n            positive_corr = top_correlations[top_correlations['Correlation'] > 0].head(5)\n            for _, row in positive_corr.iterrows():\n                st.write(f\"• {row['Feature_1']} ↔ {row['Feature_2']}: {row['Correlation']:.3f}\")\n        \n        with col2:\n            st.markdown(\"**Strongest Negative Correlations:**\")\n            negative_corr = top_correlations[top_correlations['Correlation'] < 0].head(5)\n            for _, row in negative_corr.iterrows():\n                st.write(f\"• {row['Feature_1']} ↔ {row['Feature_2']}: {row['Correlation']:.3f}\")\n    \n    def render_missing_data_analysis(self, df: pd.DataFrame) -> None:\n        \"\"\"Render missing data analysis.\"\"\"\n        \n        st.markdown(\"### 🕳️ Missing Data Analysis\")\n        \n        # Calculate missing data\n        missing_data = df.isnull().sum()\n        missing_pct = (missing_data / len(df)) * 100\n        \n        missing_df = pd.DataFrame({\n            'Column': missing_data.index,\n            'Missing_Count': missing_data.values,\n            'Missing_Percentage': missing_pct.values\n        })\n        \n        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)\n        \n        if len(missing_df) == 0:\n            st.success(\"🎉 No missing data found in the dataset!\")\n            return\n        \n        # Missing data visualization\n        fig_missing = px.bar(\n            missing_df,\n            x='Column',\n            y='Missing_Percentage',\n            title=\"Missing Data by Column\",\n            labels={'Missing_Percentage': 'Missing Percentage (%)'}\n        )\n        fig_missing.update_layout(height=400)\n        st.plotly_chart(fig_missing, use_container_width=True)\n        \n        # Missing data table\n        st.markdown(\"**Missing Data Summary:**\")\n        st.dataframe(missing_df, use_container_width=True)\n        \n        # Missing data pattern (if multiple columns have missing data)\n        if len(missing_df) > 1:\n            st.markdown(\"#### Missing Data Patterns\")\n            \n            # Create missing data pattern matrix\n            missing_cols = missing_df['Column'].tolist()\n            if len(missing_cols) <= 10:  # Only show for reasonable number of columns\n                pattern_matrix = df[missing_cols].isnull()\n                \n                # Count patterns\n                pattern_counts = pattern_matrix.value_counts()\n                \n                if len(pattern_counts) > 1:\n                    st.write(f\"Found {len(pattern_counts)} different missing data patterns:\")\n                    \n                    for i, (pattern, count) in enumerate(pattern_counts.head(5).items()):\n                        pattern_str = \", \".join([f\"{col}: {'Missing' if val else 'Present'}\" \n                                               for col, val in zip(missing_cols, pattern)])\n                        pct = (count / len(df)) * 100\n                        st.write(f\"**Pattern {i+1}** ({count:,} rows, {pct:.1f}%): {pattern_str}\")\n    \n    def render_interactive_filters(self, df: pd.DataFrame) -> pd.DataFrame:\n        \"\"\"Render interactive filters and return filtered dataframe.\"\"\"\n        \n        st.markdown(\"### 🔍 Interactive Data Filters\")\n        \n        with st.expander(\"Apply Filters\", expanded=False):\n            filtered_df = df.copy()\n            \n            # Numeric filters\n            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()\n            numeric_cols = [col for col in numeric_cols if col not in ['customerid', 'churn']]\n            \n            if numeric_cols:\n                st.markdown(\"**Numerical Filters:**\")\n                filter_cols = st.columns(2)\n                \n                for i, col in enumerate(numeric_cols[:4]):  # Limit to 4 for UI\n                    with filter_cols[i % 2]:\n                        min_val = float(df[col].min())\n                        max_val = float(df[col].max())\n                        \n                        if min_val != max_val:\n                            range_val = st.slider(\n                                f\"{col.title()}\",\n                                min_val, max_val,\n                                (min_val, max_val),\n                                key=f\"filter_{col}\"\n                            )\n                            \n                            filtered_df = filtered_df[\n                                (filtered_df[col] >= range_val[0]) & \n                                (filtered_df[col] <= range_val[1])\n                            ]\n            \n            # Categorical filters\n            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()\n            categorical_cols = [col for col in categorical_cols if col not in ['customerid', 'churn']]\n            \n            if categorical_cols:\n                st.markdown(\"**Categorical Filters:**\")\n                \n                for col in categorical_cols[:3]:  # Limit to 3 for UI\n                    unique_vals = sorted(df[col].dropna().unique())\n                    \n                    if len(unique_vals) <= 20:  # Only show filter for reasonable number of categories\n                        selected_vals = st.multiselect(\n                            f\"{col.title()}\",\n                            unique_vals,\n                            default=unique_vals,\n                            key=f\"filter_cat_{col}\"\n                        )\n                        \n                        if selected_vals:\n                            filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]\n            \n            # Show filtering results\n            original_count = len(df)\n            filtered_count = len(filtered_df)\n            \n            if filtered_count != original_count:\n                st.info(f\"Filters applied: {filtered_count:,} of {original_count:,} records shown \"\n                       f\"({filtered_count/original_count:.1%})\")\n            else:\n                st.info(\"No filters applied - showing all records\")\n        \n        return filtered_df\n    \n    def render_data_sample(self, df: pd.DataFrame) -> None:\n        \"\"\"Render data sample view.\"\"\"\n        \n        st.markdown(\"### 👀 Data Sample\")\n        \n        if df.empty:\n            st.warning(\"No data to display\")\n            return\n        \n        # Sample size selection\n        sample_size = st.select_slider(\n            \"Sample size:\",\n            options=[10, 25, 50, 100, 250, 500],\n            value=25,\n            key=\"data_sample_size\"\n        )\n        \n        # Random sample\n        if len(df) > sample_size:\n            sample_df = df.sample(n=sample_size, random_state=42)\n            st.info(f\"Showing random sample of {sample_size} records out of {len(df):,} total\")\n        else:\n            sample_df = df\n            st.info(f\"Showing all {len(df):,} records\")\n        \n        # Display sample\n        st.dataframe(sample_df, use_container_width=True)\n        \n        # Download option\n        if st.button(\"Download Sample as CSV\", key=\"download_sample\"):\n            csv = sample_df.to_csv(index=False)\n            st.download_button(\n                label=\"Click to download\",\n                data=csv,\n                file_name=f\"retention_ai_sample_{sample_size}.csv\",\n                mime=\"text/csv\"\n            )\n    \n    def render_page(self) -> None:\n        \"\"\"Render the complete data overview page.\"\"\"\n        \n        # Load data\n        df = self.load_dataset()\n        \n        if df.empty:\n            st.error(\"Unable to load data. Please ensure the database is properly configured and contains data.\")\n            return\n        \n        # Apply filters\n        filtered_df = self.render_interactive_filters(df)\n        \n        # Render all sections with filtered data\n        self.render_dataset_overview(filtered_df)\n        \n        st.markdown(\"---\")\n        self.render_churn_analysis(filtered_df)\n        \n        st.markdown(\"---\")\n        self.render_feature_distributions(filtered_df)\n        \n        st.markdown(\"---\")\n        self.render_correlation_analysis(filtered_df)\n        \n        st.markdown(\"---\")\n        self.render_missing_data_analysis(filtered_df)\n        \n        st.markdown(\"---\")\n        self.render_data_sample(filtered_df)\n\n\ndef render_data_overview_page():\n    \"\"\"Main function to render data overview page.\"\"\"\n    page = DataOverviewPage()\n    page.render_page()\n\n\nif __name__ == \"__main__\":\n    render_data_overview_page()