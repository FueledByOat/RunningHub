# base_report_generators.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.language_model_utils import LanguageModel

class BaseReportGenerator:
    def __init__(self, metrics_df, session_id, reports_dir, metadata=None, report_file_path=None):
        self.metrics_df = metrics_df if metrics_df is not None and not metrics_df.empty else pd.DataFrame()
        self.session_id = session_id
        self.metadata = metadata if metadata else {}
        self.reports_dir = reports_dir # Base directory for reports
        self.report_file_path = report_file_path # Full path to the HTML file being generated
        self.plots_sub_dir = "plots" # Standardized subdirectory for plots within reports_dir
        self.summary_data_cache = {}
        self.language_model = LanguageModel()

        self.RATING_CLASSES = {
            "optimal": "rating-optimal",
            "good": "rating-good",
            "fair": "rating-fair",
            "needs-work": "rating-needs-work",
        }
        self.PROGRESS_COLORS = {
            "optimal": "#2ecc71",
            "good": "#3498db",
            "fair": "#f39c12",
            "needs-work": "#e74c3c",
        }
        # Ensure plots directory exists when a generator is initialized
        os.makedirs(os.path.join(self.reports_dir, self.plots_sub_dir), exist_ok=True)


    def _add_html_head(self, html_content, report_title="Analysis Report"):

        # Ensure RunnerVisionLogo_transparent.png is accessible, consider making path configurable
        # or embedding as base64, or ensuring it's copied to report directory.

        html_content.extend([
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "    <meta charset='UTF-8'>",
            "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            f"    <title>{report_title}</title>",
            "    <img src='RunnerVisionLogo_transparent.png' alt='RunnerVision Logo' width='503' height='195' style='display: block; margin: 20px auto;'>",
            "    <style>",
            "        body { font-family: Arial, sans-serif; margin: 0; padding:0; background-color: #f4f4f4; color: #333; }",
            "        .container { max-width: 1200px; margin: 20px auto; background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 15px rgba(0,0,0,0.1); }",
            "        .header { text-align: center; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 1px solid #e0e0e0; }",
            "        .header h1 { color: #2c3e50; margin-bottom: 5px; } .header h2 { color: #34495e; margin-top:0; font-size: 1.2em;}",
            "        .section { margin-bottom: 30px; padding: 20px; background-color: #fdfdfd; border-radius: 8px; box-shadow: 0 0 8px rgba(0,0,0,0.07);}",
            "        .section h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-top:0; margin-bottom:20px; font-size: 1.5em; }",
            "        .section h3 { color: #34495e; margin-top: 15px; margin-bottom: 10px; font-size: 1.2em;}",
            "        .metric-box { background-color: #f9f9f9; padding: 15px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #e9e9e9; min-height: 120px; display: flex; flex-direction: column; justify-content: flex-start; }",
            "        .metric-title { font-size: 1em; color: #555; margin-bottom: 8px; font-weight: bold; }",
            "        .metric-value { font-size: 1.8em; font-weight: bold; color: #2980b9; margin-bottom: 5px; }",
            "        .metric-value-small { font-size: 1.5em; }",
            "        .metric-std { font-size: 0.8em; color: #7f8c8d; }",
            "        .row { display: flex; flex-wrap: wrap; margin-left: -10px; margin-right: -10px; }",
            "        .column { flex: 1 1 300px; padding: 10px; box-sizing: border-box; }",
            "        .chart-container { width: 100%; margin-bottom: 20px; }",
            "        img.chart { max-width: 100%; height: auto; border-radius: 5px; border: 1px solid #ddd; }",
            "        .rating { font-size: 0.95em; font-weight: bold; padding: 3px 0; }",
            f"        .rating-optimal {{ color: {self.PROGRESS_COLORS['optimal']}; }}",
            f"        .rating-good {{ color: {self.PROGRESS_COLORS['good']}; }}",
            f"        .rating-fair {{ color: {self.PROGRESS_COLORS['fair']}; }}",
            f"        .rating-needs-work {{ color: {self.PROGRESS_COLORS['needs-work']}; }}",
            "        .metric-comparison { display: flex; align-items: center; margin-top: 10px; justify-content: space-around; background-color: #f0f0f0; padding: 15px; border-radius: 5px;}",
            "        .metric-comparison-item { flex: 1; text-align: center; }",
            "        .comparison-divider { font-size: 1.2em; margin: 0 15px; color: #7f8c8d; }",
            "        .data-table { width: 100%; border-collapse: collapse; margin-top: 0px; }",
            "        .data-table th, .data-table td { border: 1px solid #ddd; padding: 10px; text-align: left; font-size:0.9em; }",
            "        .data-table th { background-color: #3498db; color: white; font-weight:bold; }",
            "        .data-table tr:nth-child(even) { background-color: #f9f9f9; }",
            "        .progress-container { width: 100%; background-color: #e0e0e0; border-radius: 5px; margin-top: 5px; height: 20px; overflow: hidden; }",
            "        .progress-bar { height: 100%; border-radius: 5px; text-align: center; line-height: 20px; color: white; font-size: 12px; transition: width 0.5s ease-in-out; }",
            f"        .progress-bar.optimal {{ background-color: {self.PROGRESS_COLORS['optimal']}; }}",
            f"        .progress-bar.good {{ background-color: {self.PROGRESS_COLORS['good']}; }}",
            f"        .progress-bar.fair {{ background-color: {self.PROGRESS_COLORS['fair']}; }}",
            f"        .progress-bar.needs-work {{ background-color: {self.PROGRESS_COLORS['needs-work']}; }}",
            "        ul { padding-left: 20px; margin-top: 5px;} li { margin-bottom: 8px; font-size: 0.95em; }",
            "        .sub-text { font-size: 0.8em; color: #666; margin-top: 5px; }",
            "    </style>",
            "</head>",
            "<body>",
            "    <div class='container'>",
            f"        <div class='header'><h1>Rear View Running Analysis</h1><h2>Session: {self.session_id}</h2></div>"
        ])
        pass

    def _add_metric_box(self, html_content, title, value_str, unit="", std_dev_str=None, rating_text=None, rating_key=None, progress_percent=None, sub_text=None):
        """Generates the HTML for a single metric box."""
        rating_class = self.RATING_CLASSES.get(rating_key, "")
        progress_bar_class = rating_key or ""

        html_content.append("            <div class='column'>")
        html_content.append("                <div class='metric-box'>")
        html_content.append(f"                    <div class='metric-title'>{title}</div>")

        main_value_display = f"{value_str}{unit}"
        if std_dev_str:
            main_value_display += f" <span class='metric-std'>&plusmn; {std_dev_str}{unit}</span>"
        html_content.append(f"                    <div class='metric-value'>{main_value_display}</div>")

        if rating_text and rating_class:
            html_content.append(f"                    <div class='rating {rating_class}'>{rating_text}</div>")

        if progress_percent is not None:
            progress_text_val = f"{progress_percent:.0f}%"
            html_content.append("                    <div class='progress-container'>")
            html_content.append(f"                        <div class='progress-bar {progress_bar_class}' style='width: {min(float(progress_percent), 100)}%;'>{progress_text_val}</div>")
            html_content.append("                    </div>")

        if sub_text:
            html_content.append(f"                    <div class='sub-text'>{sub_text}</div>")

        html_content.append("                </div>")
        html_content.append("            </div>")


    def _get_series_stats(self, df, col_name, drop_na_val="not_applicable"):

        if col_name not in df.columns or df[col_name].dropna().empty:
                        return None, None, "N/A", 0

        series = df[col_name]
        if series.dtype == 'object':
            try:
                series_numeric = pd.to_numeric(series, errors='coerce')
                if not series_numeric.isna().all(): series = series_numeric
            except Exception: pass

        if pd.api.types.is_numeric_dtype(series):
            series_cleaned = series[~pd.isna(series) & ~np.isinf(series)]
            if series_cleaned.empty: return None, None, "N/A", 0
            return series_cleaned.mean(), series_cleaned.std(), None, None
        else:
            cleaned_series = series.replace([drop_na_val, 'unknown', 'None', None, ''], pd.NA).dropna()
            if cleaned_series.empty: return None, None, "N/A", 0
            counts = cleaned_series.value_counts()
            if counts.empty: return None, None, "N/A", 0
            primary_val = str(counts.idxmax())
            percent_val = (counts.max() / counts.sum()) * 100 if counts.sum() > 0 else 0
            return None, None, primary_val, percent_val

    def _generate_session_info_section(self, html_content):

        if self.metadata:
            html_content.extend([
                "        <div class='section'>",
                "            <h2>Session Information</h2>",
                "            <div class='metric-box' style='padding: 5px 15px 15px 15px; min-height:auto;'>",
                "                <table class='data-table'>",
            ])
            for key, value in self.metadata.items():
                html_content.append(f"                    <tr><td><strong>{key.replace('_', ' ').title()}:</strong></td><td>{value}</td></tr>")
            html_content.extend([
                "                </table>",
                "            </div>",
                "        </div>",
            ])

    def _generate_main_report_structure(self, html_content, report_title_suffix=""):
        self._add_html_head(html_content, report_title=f"{self.view_name} View Running Analysis {report_title_suffix}".strip())
        self._generate_session_info_section(html_content)

        if not self.metrics_df.empty:
            summary_data = self._generate_metrics_summary_section(html_content)
            self.summary_data_cache[self.view_name.lower()] = summary_data
            self._generate_specialized_sections(html_content, summary_data) # Hook for view-specific sections
            self._generate_plots_section(html_content)
            self._generate_recommendations_section(html_content)
            self._generate_overall_assessment_section(html_content, summary_data) # If applicable to base or overridden
        else:
            html_content.append(f"<div class='section'><div class='metric-box'><p>No {self.view_name.lower()} view data loaded. Unable to generate a full report.</p></div></div>")

        html_content.extend(["     </div>", "</body>", "</html>"])


    def generate_html_file(self, output_filename_base):
        """Generates the HTML report and saves it to a file."""
        self.report_file_path = os.path.join(self.reports_dir, f"{self.session_id}_{output_filename_base}_{self.view_name.lower()}_report.html")
        self.report_file_name = f"{self.session_id}_{output_filename_base}_{self.view_name.lower()}_report.html"
        html_content = []
        self._generate_main_report_structure(html_content) # Call the main structure builder

        try:
            with open(self.report_file_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(html_content))
            print(f"✅ {self.view_name} view report successfully generated: {self.report_file_path}")
        except IOError as e:
            print(f"❌ Error writing {self.view_name.lower()} view report file: {e}")
        except Exception as e:
            print(f"❌ An unexpected error occurred during {self.view_name.lower()} report generation: {e}")
        return self.report_file_path


    # Abstract methods to be implemented by subclasses
    @property
    def view_name(self): # e.g., "Side", "Rear"
        raise NotImplementedError

    def _generate_metrics_summary_section(self, html_content):
        raise NotImplementedError # To be implemented by Side/Rear specific generators

    def _generate_specialized_sections(self, html_content, summary_data):
        """Hook for view-specific sections beyond basic summary and plots."""
        pass # Optional: implement in subclasses if needed

    def _generate_plots_section(self, html_content):
        raise NotImplementedError

    def _generate_recommendations_section(self, html_content):
        raise NotImplementedError

    def _generate_overall_assessment_section(self, html_content, summary_data):
        # This might be common enough for base, or can be overridden
        pass