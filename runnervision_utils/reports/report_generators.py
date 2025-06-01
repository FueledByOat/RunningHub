# report_generators.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class BaseReportGenerator:
    def __init__(self, metrics_df, session_id, reports_dir, metadata=None, report_file_path=None):
        self.metrics_df = metrics_df if metrics_df is not None and not metrics_df.empty else pd.DataFrame()
        self.session_id = session_id
        self.metadata = metadata if metadata else {}
        self.reports_dir = reports_dir # Base directory for reports
        self.report_file_path = report_file_path # Full path to the HTML file being generated
        self.plots_sub_dir = "plots" # Standardized subdirectory for plots within reports_dir
        self.summary_data_cache = {}

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
                    f"       .rating-optimal {{ color: {self.PROGRESS_COLORS['optimal']}; }}",
                    f"       .rating-good {{ color: {self.PROGRESS_COLORS['good']}; }}",
                    f"       .rating-fair {{ color: {self.PROGRESS_COLORS['fair']}; }}",
                    f"       .rating-needs-work {{ color: {self.PROGRESS_COLORS['needs-work']}; }}",
                    "        .metric-comparison { display: flex; align-items: center; margin-top: 10px; justify-content: space-around; background-color: #f0f0f0; padding: 15px; border-radius: 5px;}",
                    "        .metric-comparison-item { flex: 1; text-align: center; }",
                    "        .comparison-divider { font-size: 1.2em; margin: 0 15px; color: #7f8c8d; }",
                    "        .data-table { width: 100%; border-collapse: collapse; margin-top: 0px; }",
                    "        .data-table th, .data-table td { border: 1px solid #ddd; padding: 10px; text-align: left; font-size:0.9em; }",
                    "        .data-table th { background-color: #3498db; color: white; font-weight:bold; }",
                    "        .data-table tr:nth-child(even) { background-color: #f9f9f9; }",
                    "        .progress-container { width: 100%; background-color: #e0e0e0; border-radius: 5px; margin-top: 5px; height: 20px; overflow: hidden; }",
                    "        .progress-bar { height: 100%; border-radius: 5px; text-align: center; line-height: 20px; color: white; font-size: 12px; transition: width 0.5s ease-in-out; }",
                    f"       .progress-bar.optimal {{ background-color: {self.PROGRESS_COLORS['optimal']}; }}",
                    f"       .progress-bar.good {{ background-color: {self.PROGRESS_COLORS['good']}; }}",
                    f"       .progress-bar.fair {{ background-color: {self.PROGRESS_COLORS['fair']}; }}",
                    f"       .progress-bar.needs-work {{ background-color: {self.PROGRESS_COLORS['needs-work']}; }}",
                    "        ul { padding-left: 20px; margin-top: 5px;} li { margin-bottom: 8px; font-size: 0.95em; }",
                    "        .sub-text { font-size: 0.8em; color: #666; margin-top: 5px; }",
                    "    </style>",
                    "</head>",
                    "<body>",
                    "    <div class='container'>",
                    f"        <div class='header'><h1>Rear View Running Analysis</h1><h2>Session: {self.session_id}</h2></div>"
                ])
        pass

    def _add_metric_box(self, html_content, title, value_str, unit="", std_dev_str=None, rating_text=None, rating_class=None, progress_percent=None, progress_bar_class_key=None, sub_text=None):
        # ... (common metric box HTML logic)
        html_content.append("            <div class='column'>")
        html_content.append("                <div class='metric-box'>")
        html_content.append(f"                    <div class='metric-title'>{title}</div>")
        main_value_display = f"{value_str}{unit}"
        if std_dev_str:
            main_value_display += f" <span class='metric-std'>&plusmn; {std_dev_str}{unit}</span>"
        html_content.append(f"                    <div class='metric-value'>{main_value_display}</div>")
        if rating_text and rating_class:
            html_content.append(f"                    <div class='rating {rating_class}'>{rating_text}</div>")
        if progress_percent is not None and progress_bar_class_key:
            progress_text_val = f"{progress_percent:.0f}%"
            # Customize progress bar text if needed for specific metrics
            # if title == "Specific Metric": progress_text_val = f"{value_str}"
            html_content.append("                    <div class='progress-container'>")
            html_content.append(f"                        <div class='progress-bar {progress_bar_class_key}' style='width: {min(float(progress_percent), 100)}%;'>{progress_text_val}</div>")
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


class SideViewReportGenerator(BaseReportGenerator):
    @property
    def view_name(self):
        return "Side"

    def _generate_metrics_summary_section(self, html_content):
        # ... (specific logic from current SideReportGenerator._generate_metrics_summary_section)
        # Call self._add_metric_box, self._get_series_stats(self.metrics_df, ...)
        if self.metrics_df.empty: # Add this check
            html_content.append("<div class='section'><h2>Side Metrics Summary</h2><div class='metric-box'><p>No side view metrics data available.</p></div></div>")
            return {}
        # ... existing logic using self.metrics_df ...
        summary_metrics_data = {} # populate this

        def add_summary_metric(col_name, title, unit="", val_format="{:.1f}", is_categorical=False, cat_sub_text_format="({percent:.1f}% dominance)"):
            mean, std, primary, percent = self._get_series_stats(self.metrics_df, col_name)
            
            if is_categorical:
                summary_metrics_data[f"{col_name}_primary"] = primary
                summary_metrics_data[f"{col_name}_percent"] = percent
                sub_text = cat_sub_text_format.format(percent=percent) if primary != "N/A" else None
                self._add_metric_box(html_content, title, primary if primary else "N/A", sub_text=sub_text)
            else:
                summary_metrics_data[f"{col_name}_mean"] = mean
                summary_metrics_data[f"{col_name}_std"] = std
                value_str = val_format.format(mean) if mean is not None else "N/A"
                std_str = "{:.1f}".format(std) if std is not None else None
                self._add_metric_box(html_content, title, value_str, unit=unit, std_dev_str=std_str)

        add_summary_metric('strike_pattern', "Foot Strike Pattern", is_categorical=True)
        add_summary_metric('foot_landing_position_category', "Foot Landing Pattern", is_categorical=True)
        add_summary_metric('trunk_angle_degrees', "Trunk Angle", unit="°")
        add_summary_metric('knee_angle_left', "Left Knee Angle (Avg)", unit="°")
        add_summary_metric('knee_angle_right', "Right Knee Angle (Avg)", unit="°")

        # Stance Phase Percentage (calculated differently in original, special handling)
        if 'stance_phase_detected' in self.metrics_df.columns:
            stance_series = self.metrics_df['stance_phase_detected'].dropna().astype(bool)
            if not stance_series.empty:
                spp = (stance_series.sum() / len(stance_series) * 100) if len(stance_series) > 0 else 0
                summary_metrics_data['stance_phase_percent'] = spp
                self._add_metric_box(html_content, "Stance Phase", f"{spp:.1f}", unit="% of gait cycle")
            else: self._add_metric_box(html_content, "Stance Phase", "N/A")
        else: self._add_metric_box(html_content, "Stance Phase", "N/A")
        
        add_summary_metric('stride_length_cm', "Stride Length", unit=" cm")
        add_summary_metric('normalized_stride_length', "Normalized Stride", unit=" × height", val_format="{:.2f}")
        add_summary_metric('strike_landing_stiffness', "Landing Stiffness", is_categorical=True)
        add_summary_metric('vertical_oscillation_cm', "Vertical Oscillation", unit=" cm")
        add_summary_metric('avg_contact_time_ms', "Ground Contact Time", unit=" ms", val_format="{:.0f}")
        add_summary_metric('cadence', "Cadence", unit=" spm", val_format="{:.0f}")

        html_content.extend(["            </div>", "        </div>"])
        return summary_metrics_data


    def _generate_bilateral_comparison_section(self, html_content, summary_data):
        if self.metrics_df.empty: return

        left_avg = summary_data.get('knee_angle_left_mean')
        right_avg = summary_data.get('knee_angle_right_mean')

        if left_avg is None or right_avg is None or pd.isna(left_avg) or pd.isna(right_avg):
            html_content.append("<div class='section'><h2>Bilateral Comparison</h2><div class='metric-box'><p>Knee angle data not available for comparison.</p></div></div>")
            return

        html_content.extend([
            "        <div class='section'>",
            "            <h2>Bilateral Comparison</h2>",
            "            <div class='metric-box' style='padding:20px;'>",
            "                <h3>Knee Angle Symmetry</h3>",
            "                <div class='metric-comparison'>",
            "                    <div class='metric-comparison-item'>",
            f"                        <div>Left Knee</div><div class='metric-value metric-value-small'>{left_avg:.1f}°</div>",
            "                    </div>",
            "                    <div class='comparison-divider'>vs</div>",
            "                    <div class='metric-comparison-item'>",
            f"                        <div>Right Knee</div><div class='metric-value metric-value-small'>{right_avg:.1f}°</div>",
            "                    </div>",
            "                </div>"
        ])

        diff_abs = abs(left_avg - right_avg)
        denominator = (left_avg + right_avg) / 2
        diff_percent = (diff_abs / denominator) * 100 if denominator != 0 else 0
        summary_data['knee_symmetry_diff_percent'] = diff_percent
        
        rating_text, rating_class = "Needs improvement", self.RATING_CLASSES["needs-work"]
        if diff_percent < 5: rating_text, rating_class = "Excellent symmetry", self.RATING_CLASSES["optimal"]
        elif diff_percent < 10: rating_text, rating_class = "Good symmetry", self.RATING_CLASSES["good"]
        elif diff_percent < 15: rating_text, rating_class = "Fair symmetry", self.RATING_CLASSES["fair"]

        html_content.extend([
            f"                <div style='text-align:center; margin-top:15px; font-size:0.95em;'>Difference: {diff_abs:.1f}° ({diff_percent:.1f}%)</div>",
            f"                <div class='rating {rating_class}' style='text-align:center; margin-top:5px;'>{rating_text}</div>",
            "            </div>",
            "        </div>"
        ])

    def _generate_arm_swing_section(self, html_content, summary_data):
        if self.metrics_df.empty: return
        
        has_amp = 'arm_swing_amplitude' in self.metrics_df.columns
        has_sym = 'arm_swing_symmetry' in self.metrics_df.columns

        if not (has_amp or has_sym):
            html_content.append("<div class='section'><h2>Arm Swing Analysis</h2><div class='metric-box'><p>Arm swing data not available.</p></div></div>")
            return

        html_content.extend([
            "        <div class='section'>",
            "            <h2>Arm Swing Analysis</h2>",
            "            <div class='row'>"
        ])

        # Arm Swing Amplitude
        if has_amp:
            amp_mean, _, _, _ = self._get_series_stats(self.metrics_df, 'arm_swing_amplitude')
            summary_data['arm_swing_amplitude_mean'] = amp_mean
            if amp_mean is not None and not pd.isna(amp_mean):
                amp_rating_text, amp_rating_class, amp_rating_key = "Limited", self.RATING_CLASSES["needs-work"], "needs-work"
                if amp_mean > 45: amp_rating_text, amp_rating_class, amp_rating_key = "Optimal", self.RATING_CLASSES["optimal"], "optimal"
                elif amp_mean > 35: amp_rating_text, amp_rating_class, amp_rating_key = "Good", self.RATING_CLASSES["good"], "good"
                elif amp_mean > 25: amp_rating_text, amp_rating_class, amp_rating_key = "Fair", self.RATING_CLASSES["fair"], "fair"
                
                progress_val_amp = min(amp_mean / 60 * 100, 100) # Assume 60 deg is a ref max for progress bar display

                self._add_metric_box(html_content, "Arm Swing Amplitude", f"{amp_mean:.1f}", unit="°",
                                    rating_text=amp_rating_text, rating_class=amp_rating_class,
                                    progress_percent=progress_val_amp, progress_bar_class_key=amp_rating_key,
                                    sub_text="Avg. peak-to-peak amplitude.")
            else:
                self._add_metric_box(html_content, "Arm Swing Amplitude", "N/A")
        
        # Arm Swing Symmetry
        if has_sym:
            sym_series = self.metrics_df['arm_swing_symmetry'].dropna()
            arm_swing_sym_percent = 0
            if not sym_series.empty:
                symmetrical_counts = sym_series[sym_series == 'symmetrical'].count()
                arm_swing_sym_percent = (symmetrical_counts / len(sym_series)) * 100 if len(sym_series) > 0 else 0
            summary_data['arm_swing_symmetry_percent'] = arm_swing_sym_percent
            
            sym_rating_text, sym_rating_class, sym_rating_key = "Asymmetrical", self.RATING_CLASSES["needs-work"], "needs-work"
            if arm_swing_sym_percent > 90: sym_rating_text, sym_rating_class, sym_rating_key = "Excellent symmetry", self.RATING_CLASSES["optimal"], "optimal"
            elif arm_swing_sym_percent > 80: sym_rating_text, sym_rating_class, sym_rating_key = "Good symmetry", self.RATING_CLASSES["good"], "good"
            elif arm_swing_sym_percent > 70: sym_rating_text, sym_rating_class, sym_rating_key = "Fair symmetry", self.RATING_CLASSES["fair"], "fair"
            
            self._add_metric_box(html_content, "Arm Swing Symmetry", f"{arm_swing_sym_percent:.1f}", unit="%",
                                rating_text=sym_rating_text, rating_class=sym_rating_class,
                                progress_percent=arm_swing_sym_percent, progress_bar_class_key=sym_rating_key,
                                sub_text="% of time symmetrical.")
        
        html_content.extend(["            </div>", "        </div>"])


    def _generate_stride_analysis_section(self, html_content, summary_data):
        if self.metrics_df.empty: return

        has_sl = 'stride_length_cm_mean' in summary_data
        has_cad = 'cadence_mean' in summary_data or 'stride_frequency_mean' in summary_data

        if not (has_sl or has_cad):
            html_content.append("<div class='section'><h2>Stride Analysis</h2><div class='metric-box'><p>Stride data not available.</p></div></div>")
            return

        html_content.extend([
            "        <div class='section'>",
            "            <h2>Stride Analysis</h2>",
            "            <div class='row'>"
        ])
        
        # Stride Length & Normalized Stride
        if has_sl:
            stride_length_mean = summary_data.get('stride_length_cm_mean')
            norm_stride_mean = summary_data.get('normalized_stride_length_mean')
            
            if stride_length_mean is not None and not pd.isna(stride_length_mean):
                sl_sub_text, sl_rating_text, sl_rating_class = "", None, None
                if norm_stride_mean is not None and not pd.isna(norm_stride_mean):
                    sl_sub_text = f"Normalized: {norm_stride_mean:.2f} × height"
                    if 1.2 <= norm_stride_mean <= 1.4: sl_rating_text, sl_rating_class = "Optimal Range", self.RATING_CLASSES["optimal"]
                    elif (1.0 <= norm_stride_mean < 1.2) or (1.4 < norm_stride_mean <= 1.6): sl_rating_text, sl_rating_class = "Good Range", self.RATING_CLASSES["good"]
                    else: sl_rating_text, sl_rating_class = "Outside Optimal", self.RATING_CLASSES["needs-work"]
                
                self._add_metric_box(html_content, "Stride Length", f"{stride_length_mean:.1f}", unit=" cm",
                                    rating_text=sl_rating_text, rating_class=sl_rating_class, sub_text=sl_sub_text)
            else:
                self._add_metric_box(html_content, "Stride Length", "N/A")

        # Cadence
        if has_cad:
            cadence_mean = summary_data.get('cadence_mean') # Garmin's SPM is preferred
            if (cadence_mean is None or pd.isna(cadence_mean)) and 'stride_frequency_mean' in summary_data:
                stride_freq_mean = summary_data.get('stride_frequency_mean') # strides/min
                if stride_freq_mean is not None and not pd.isna(stride_freq_mean):
                    cadence_mean = stride_freq_mean * 2
                    summary_data['cadence_mean_calculated'] = cadence_mean # Store if we had to calculate it

            if cadence_mean is not None and not pd.isna(cadence_mean):
                cad_rating_text, cad_rating_class = "Suboptimal Cadence", self.RATING_CLASSES["needs-work"]
                if 170 <= cadence_mean <= 190: cad_rating_text, cad_rating_class = "Optimal Cadence", self.RATING_CLASSES["optimal"]
                elif (160 <= cadence_mean < 170) or (190 < cadence_mean <= 200): cad_rating_text, cad_rating_class = "Good Cadence", self.RATING_CLASSES["good"]
                
                self._add_metric_box(html_content, "Cadence", f"{cadence_mean:.0f}", unit=" spm",
                                    rating_text=cad_rating_text, rating_class=cad_rating_class, sub_text="Steps Per Minute.")
            else:
                self._add_metric_box(html_content, "Cadence", "N/A")

        html_content.extend(["            </div>", "        </div>"])


    def _generate_ground_contact_section(self, html_content, summary_data):
        if self.metrics_df.empty: 
            return

        has_stance_dist = 'stance_foot' in self.metrics_df.columns and 'stance_phase_detected' in self.metrics_df.columns
        has_stiffness = 'strike_landing_stiffness_primary' in summary_data
        
        if not (has_stance_dist or has_stiffness):
            html_content.append("<div class='section'><h2>Ground Contact Analysis</h2><div class='metric-box'><p>Ground contact data not available.</p></div></div>")
            return

        html_content.extend([
            "        <div class='section'>",
            "            <h2>Ground Contact Analysis</h2>",
            "            <div class='row'>"
        ])

        # Stance Phase Distribution
        if has_stance_dist:
            stance_foot_series = self.metrics_df['stance_foot'].replace(['not_applicable', 'unknown', None, ''], pd.NA).dropna()
            if not stance_foot_series.empty:
                stance_counts = stance_foot_series.value_counts()
                total_stance_frames = stance_counts.sum()

                if total_stance_frames > 0:
                    html_content.append("            <div class='column'>")
                    html_content.append("                <div class='metric-box'>")
                    html_content.append("                    <div class='metric-title'>Stance Phase Distribution</div>")
                    
                    for foot, count in stance_counts.items():
                        if str(foot).lower() in ['left', 'right']:
                            percentage = count / total_stance_frames * 100
                            bar_color_class_key = "good" if foot.lower() == 'left' else "fair" # Example colors
                            
                            html_content.append(f"                    <div class='sub-text' style='margin-bottom: 2px;'>{str(foot).capitalize()} Foot: {percentage:.1f}% ({count} frames)</div>")
                            html_content.append("                    <div class='progress-container' style='height:15px; margin-bottom:8px;'>")
                            html_content.append(f"                        <div class='progress-bar {bar_color_class_key}' style='width: {percentage}%;'></div>")
                            html_content.append("                    </div>")
                    html_content.append("                </div>")
                    html_content.append("            </div>")

        # Landing Stiffness
        if has_stiffness:
            ls_primary = summary_data.get('strike_landing_stiffness_primary', "N/A")
            ls_percent = summary_data.get('strike_landing_stiffness_percent', 0)

            if ls_primary != "N/A":
                stiffness_score = ls_percent / 10.0 # Scale 0-100% dominance to 0-10 score
                summary_data['landing_stiffness_score_calculated'] = stiffness_score

                stiffness_rating_text, stiffness_class, stiffness_rating_key = "Suboptimal", self.RATING_CLASSES["needs-work"], "needs-work"
                if 4 <= stiffness_score <= 6: stiffness_rating_text, stiffness_class, stiffness_rating_key = "Optimal", self.RATING_CLASSES["optimal"], "optimal"
                elif (3 <= stiffness_score < 4) or (6 < stiffness_score <= 7): stiffness_rating_text, stiffness_class, stiffness_rating_key = "Good", self.RATING_CLASSES["good"], "good"
                elif (2 <= stiffness_score < 3) or (7 < stiffness_score <= 8): stiffness_rating_text, stiffness_class, stiffness_rating_key = "Fair", self.RATING_CLASSES["fair"], "fair"
                
                progress_val_stiff = stiffness_score * 10 # For 0-100% bar

                self._add_metric_box(html_content, "Landing Stiffness", f"{stiffness_score:.1f}",
                                    rating_text=f"{stiffness_rating_text} stiffness", rating_class=stiffness_class,
                                    progress_percent=progress_val_stiff, progress_bar_class_key=stiffness_rating_key,
                                    sub_text=f"Primary: {ls_primary} ({ls_percent:.1f}%). Score based on dominance. 0=Soft, 10=Stiff.")
            else:
                self._add_metric_box(html_content, "Landing Stiffness", "N/A")
        
        html_content.extend(["            </div>", "        </div>"])


    def _generate_specialized_sections(self, html_content, summary_data):
        self._generate_bilateral_comparison_section(html_content, summary_data)
        self._generate_arm_swing_section(html_content, summary_data)
        self._generate_stride_analysis_section(html_content, summary_data)
        self._generate_ground_contact_section(html_content, summary_data)

    def _generate_plots_section(self, html_content):
        # ... (logic from current SideReportGenerator._generate_plots_section)
        # This should call a plotting utility, e.g.,
        # plot_files = plotting_utils.save_side_view_plots(self.metrics_df, self.session_id, os.path.join(self.reports_dir, self.plots_sub_dir))
        # Then add plot_files to HTML
        if self.metrics_df.empty: return

        plot_files = []
        if hasattr(self, '_save_side_metric_plots') and callable(self._save_side_metric_plots):
            plot_files = self._save_side_metric_plots()

        if not plot_files:
            html_content.append("<div class='section'><h2>Running Metrics Visualization</h2><div class='metric-box'><p>No plots generated or available.</p></div></div>")
            return

        html_content.extend(["        <div class='section'>", "            <h2>Running Metrics Visualization</h2>"])
        
        for i, plot_file in enumerate(plot_files):
            if i % 2 == 0: # Start new row
                if i > 0: html_content.append("            </div>")
                html_content.append("            <div class='row'>")
            
            base_dir = os.path.dirname(self.report_file_path) if self.report_file_path else "."
            rel_path = plot_file # Default to using the path as is
            try:
                # Create relative path only if plot_file is absolute and shares common prefix with base_dir
                if os.path.isabs(plot_file) and os.path.commonprefix([plot_file, os.path.abspath(base_dir)]):
                    rel_path = os.path.relpath(plot_file, base_dir)
                elif not os.path.isabs(plot_file): # if plot_file is already relative, assume it's correct
                    pass 
            except ValueError: pass # If on different drives (Windows) or other relpath error

            html_content.extend([
                "                <div class='column'>",
                "                    <div class='chart-container'>",
                f"                        <img src='{rel_path}' alt='Metrics Plot {i+1}' class='chart'>",
                "                    </div>",
                "                </div>"
            ])
        
        if plot_files: html_content.append("            </div>") # Close final row
        html_content.append("        </div>")


    def _generate_recommendations_section(self, html_content):
        # ... (logic from current SideReportGenerator._generate_recommendations_section)
        # recommendations = recommendation_engine.generate_side_view_recommendations(self.metrics_df, self.summary_data_cache.get('side'))
        if self.metrics_df.empty: return
            
        recommendations = []
        if hasattr(self, '_generate_side_recommendations') and callable(self._generate_side_recommendations):
            recommendations = self._generate_side_recommendations()

        html_content.append("<div class='section'><h2>Gait Analysis & Recommendations</h2><div class='metric-box' style='min-height:auto; padding: 5px 15px 15px 15px;'>")
        if recommendations:
            html_content.append("<h3>Form Recommendations</h3><ul>")
            for rec in recommendations:
                html_content.append(f"    <li>{rec}</li>")
            html_content.append("</ul>")
        else:
            html_content.append("<p>No specific recommendations generated at this time. Focus on maintaining consistent form.</p>")
        html_content.append("</div></div>")

    def _generate_overall_assessment_section(self, html_content, summary_data):
        # ... (logic from current SideReportGenerator._generate_overall_assessment_section)
        if self.metrics_df.empty or not summary_data:
            html_content.append("<div class='section'><h2>Overall Assessment</h2><div class='metric-box'><p>Insufficient data for an overall assessment.</p></div></div>")
            return

        score_components = []
        strengths, improvements = [], []

        # Foot Strike
        fs_primary = summary_data.get('strike_pattern_primary')
        if fs_primary == 'midfoot': score_components.append(10); strengths.append("Efficient midfoot strike pattern.")
        elif fs_primary == 'forefoot': score_components.append(8); strengths.append("Dynamic forefoot strike (monitor calf/Achilles).")
        elif fs_primary == 'heel': score_components.append(6); improvements.append("Consider transitioning towards a midfoot strike to potentially reduce braking forces.")
        
        # Trunk Angle (Original logic: assess `100 - trunk_mean`)
        trunk_mean = summary_data.get('trunk_angle_degrees_mean')
        if trunk_mean is not None and not pd.isna(trunk_mean):
            # This interpretation assumes trunk_angle_degrees is forward lean from vertical.
            # E.g. 5-10 degrees is optimal. The `100-X` logic in original implies a different scale.
            # For consistency with original's scoring, let's map it:
            # If original `trunk_angle_degrees` was, say, angle from horizontal (80-85 optimal lean),
            # then `100 - (angle_from_horizontal)` -> 100-80=20, 100-85=15. This doesn't fit 85-95.
            # Let's assume `trunk_mean` *is* the value that the original `100-X` was targeting.
            # Or, more directly, optimal lean is slight. 0-5 deg too upright/back, 5-10 optimal, >10 too much.
            # Original: 85 <= (100-trunk_mean) <= 95  => 5 <= trunk_mean <= 15 for score 10.
            if 5 <= trunk_mean <= 15: # Assuming trunk_mean is fwd lean in degrees
                score_components.append(10); strengths.append(f"Good trunk posture (lean: {trunk_mean:.1f}°).")
            elif (0 <= trunk_mean < 5) or (15 < trunk_mean <= 20):
                score_components.append(8)
                if trunk_mean < 5 : improvements.append(f"Slightly upright posture (lean: {trunk_mean:.1f}°), consider a bit more forward lean from the ankles.")
                else: improvements.append(f"Slightly excessive lean (lean: {trunk_mean:.1f}°), ensure it's from ankles not waist.")
            else:
                score_components.append(5)
                if trunk_mean < 0 : improvements.append(f"Avoid backward lean (lean: {trunk_mean:.1f}°). Engage core for slight forward lean.")
                elif trunk_mean > 20 : improvements.append(f"Reduce excessive forward lean (lean: {trunk_mean:.1f}°). Focus on posture.")

        # Knee Symmetry
        knee_diff = summary_data.get('knee_symmetry_diff_percent')
        if knee_diff is not None:
            if knee_diff < 5: score_components.append(10); strengths.append("Excellent bilateral knee symmetry.")
            elif knee_diff < 10: score_components.append(8) # Good, no specific comment unless very borderline
            else: # >=10
                score_components.append(6 if knee_diff < 15 else 4)
                improvements.append(f"Address knee angle asymmetry ({knee_diff:.1f}% difference). Consider strength/flexibility imbalances.")

        # Arm Swing Amplitude
        arm_amp = summary_data.get('arm_swing_amplitude_mean')
        if arm_amp is not None and not pd.isna(arm_amp):
            if arm_amp > 45: score_components.append(10); strengths.append("Optimal arm swing amplitude, promoting efficiency.")
            elif arm_amp > 35: score_components.append(8)
            else: # <35
                score_components.append(6 if arm_amp > 25 else 4)
                improvements.append("Increase arm swing amplitude for better balance and propulsive force.")

        # Cadence
        cadence = summary_data.get('cadence_mean') or summary_data.get('cadence_mean_calculated')
        if cadence is not None and not pd.isna(cadence):
            if 170 <= cadence <= 190: score_components.append(10); strengths.append(f"Optimal cadence ({cadence:.0f} spm) for good turnover.")
            elif (160 <= cadence < 170) or (190 < cadence <= 200): score_components.append(8)
            else:
                score_components.append(6)
                if cadence < 160: improvements.append(f"Increase cadence (currently {cadence:.0f} spm) to potentially reduce overstriding and impact.")
                else: improvements.append(f"Cadence ({cadence:.0f} spm) is high; ensure it's comfortable and efficient for your pace, not overly choppy.")
        
        overall_score, perf_cat, perf_class, perf_key = 0, "N/A", self.RATING_CLASSES["fair"], "fair"
        if score_components:
            overall_score = (sum(score_components) / (len(score_components) * 10)) * 100
            if overall_score >= 90: perf_cat, perf_class, perf_key = "Excellent", self.RATING_CLASSES["optimal"], "optimal"
            elif overall_score >= 80: perf_cat, perf_class, perf_key = "Good", self.RATING_CLASSES["good"], "good"
            elif overall_score >= 70: perf_cat, perf_class, perf_key = "Fair", self.RATING_CLASSES["fair"], "fair"
            else: perf_cat, perf_class, perf_key = "Needs Improvement", self.RATING_CLASSES["needs-work"], "needs-work"

        html_content.extend([
            "        <div class='section'><h2>Overall Assessment</h2>",
            "            <div class='metric-box' style='padding:20px;'>",
            "                <div style='text-align: center;'>",
            f"                    <div class='metric-value' style='font-size: 2.5em;'>{overall_score:.0f}/100</div>",
            f"                    <div class='rating {perf_class}' style='font-size:1.3em;'>{perf_cat}</div>",
            "                </div>",
            "                <div class='progress-container' style='height: 30px; margin: 20px auto; max-width: 90%;'>",
            f"                    <div class='progress-bar {perf_key}' style='width: {max(0,min(overall_score,100))}%; height: 30px; line-height:30px; font-size:1em;'>{overall_score:.0f}%</div>",
            "                </div>"
        ])
        html_content.append("<h3>Key Strengths:</h3><ul>" + ("".join(f"<li>{s}</li>" for s in strengths[:3]) if strengths else "<li>Continue to build on consistent training.</li>") + "</ul>")
        html_content.append("<h3>Areas for Improvement:</h3><ul>" + ("".join(f"<li>{i}</li>" for i in improvements[:3]) if improvements else "<li>Great job! Focus on consistency and gradual progression.</li>") + "</ul>")
        html_content.extend(["            </div></div>"])

    def _generate_side_recommendations(self):
        """Generate running form recommendations based on metrics."""
        if self.metrics_df is None:
            return []
        
        recommendations = []
        
        # Analyze foot strike pattern
        foot_strike_counts = self.metrics_df['strike_pattern'].value_counts().drop("not_applicable", errors='ignore')
        primary_foot_strike = foot_strike_counts.idxmax() if not foot_strike_counts.empty else None
        
        if primary_foot_strike == 'heel':
            recommendations.append(
                "Your running shows a predominant heel strike pattern. Consider working on a more midfoot landing "
                "to reduce impact forces and improve efficiency. Try shortening your stride slightly and increasing cadence."
            )
        elif primary_foot_strike == 'forefoot':
            recommendations.append(
                "You're landing primarily on your forefoot, which is efficient but can place more stress on your calves "
                "and Achilles. Make sure you're allowing your heel to drop slightly after initial contact for better shock absorption."
            )
        elif primary_foot_strike == 'midfoot':
            recommendations.append(
                "Your midfoot strike pattern is generally efficient and helps distribute impact forces well. "
                "Maintain this landing pattern while focusing on a quick cadence."
            )
        
        # Analyze trunk angle
        trunk_angle_mean = self.metrics_df['trunk_angle_degrees'].mean()
        if 100 - trunk_angle_mean < 85:
            recommendations.append(
                f"Your average trunk angle of {trunk_angle_mean:.1f}° indicates excessive forward lean. "
                "Work on maintaining a more upright posture by engaging your core muscles and focusing on running tall."
            )
        elif 100 - trunk_angle_mean > 95:
            recommendations.append(
                f"Your average trunk angle of {trunk_angle_mean:.1f}° indicates a backward lean. "
                "Focus on a slight forward lean from the ankles rather than the waist to improve running economy."
            )
        else:
            recommendations.append(
                f"Your average trunk angle of {trunk_angle_mean:.1f}° is within the optimal range. "
                "Continue maintaining this posture for optimal running economy."
            )
        
        # Analyze knee angles and symmetry
        if 'knee_angle_left' in self.metrics_df.columns and 'knee_angle_right' in self.metrics_df.columns:
            left_knee_mean = self.metrics_df['knee_angle_left'].mean()
            right_knee_mean = self.metrics_df['knee_angle_right'].mean()
            diff = abs(left_knee_mean - right_knee_mean)
            diff_percent = diff / ((left_knee_mean + right_knee_mean) / 2) * 100
            
            if diff_percent > 10:
                recommendations.append(
                    f"There's a {diff_percent:.1f}% difference between your left and right knee angles, which may indicate muscular imbalances. "
                    "Consider targeted strength training to address this asymmetry and reduce injury risk."
                )
        
        # Analyze arm swing
        if 'arm_swing_amplitude' in self.metrics_df.columns:
            arm_swing_mean = self.metrics_df['arm_swing_amplitude'].mean()
            if arm_swing_mean < 30:
                recommendations.append(
                    f"Your arm swing amplitude of {arm_swing_mean:.1f}° is limited. "
                    "Try to increase your arm movement with a 90° bend at the elbow, allowing your arms to swing naturally from your shoulders."
                )
        
        if 'arm_swing_symmetry' in self.metrics_df.columns:
            # arm_symmetry = self.metrics_df['arm_swing_symmetry'].mean()

            arm_swing_symmetry_counts = self.metrics_df['arm_swing_symmetry'].value_counts()['symmetrical'] if 'arm_swing_symmetry' in self.metrics_df.columns else None
            arm_symmetry = (arm_swing_symmetry_counts / len(self.metrics_df['arm_swing_symmetry'])) * 100
            if arm_symmetry < 75:
                recommendations.append(
                    f"Your arm swing shows asymmetry at {arm_symmetry:.1f}% similarity. "
                    "Focus on balanced arm movement to improve overall running efficiency and reduce rotation."
                )
        
        # Analyze foot landing position
        foot_landing_counts = self.metrics_df['foot_landing_position_category'].value_counts().drop("not_applicable", errors='ignore')
        primary_foot_landing = foot_landing_counts.idxmax() if not foot_landing_counts.empty else None
        
        if primary_foot_landing == 'ahead':
            recommendations.append(
                "Your feet are landing predominantly ahead of your center of mass, which may increase braking forces. "
                "Work on landing with your foot closer to your center of gravity by increasing cadence and focusing on pulling your foot up quickly after contact."
            )
        elif primary_foot_landing == 'behind':
            recommendations.append(
                "Your feet are landing predominantly behind your center of mass, which is unusual. "
                "This may indicate overstriding or form compensation. Focus on a more natural foot landing pattern."
            )
        
        # Analyze stride metrics
        if 'stride_frequency' in self.metrics_df.columns:
            stride_freq = self.metrics_df['stride_frequency'].mean()
            cadence = stride_freq * 2  # Convert to steps per minute
            
            if cadence < 160:
                recommendations.append(
                    f"Your current cadence of {cadence:.1f} steps/minute is relatively low. "
                    "Consider gradually increasing your cadence to 170-180 steps/minute to reduce overstriding and impact forces."
                )
            elif cadence > 200:
                recommendations.append(
                    f"Your current cadence of {cadence:.1f} steps/minute is quite high. "
                    "While this can be efficient for sprinting, for distance running you might benefit from a slightly lower cadence of 170-190 steps/minute."
                )
        
        # Analyze landing stiffness
        if 'strike_landing_stiffness' in self.metrics_df.columns:
            # stiffness = self.metrics_df['strike_landing_stiffness'].mean()
            landing_stiffness_counts = self.metrics_df['strike_landing_stiffness'].value_counts().drop("not_applicable", errors='ignore')
            primary_landing_stiffness = landing_stiffness_counts.idxmax() if not landing_stiffness_counts.empty else "N/A"
            stiffness = (landing_stiffness_counts.max() / landing_stiffness_counts.sum() * 10) if not landing_stiffness_counts.empty else 0
            
            if stiffness < 3:
                recommendations.append(
                    f"Your landing stiffness rating of {stiffness:.1f}/10 indicates a very soft landing. "
                    "While this reduces impact forces, it may reduce energy return. Consider developing more reactive strength for better running economy."
                )
            elif stiffness > 7:
                recommendations.append(
                    f"Your landing stiffness rating of {stiffness:.1f}/10 indicates a very stiff landing. "
                    "This increases impact forces and may lead to injuries. Focus on softer landings with a slightly bent knee at contact."
                )
        
        return recommendations[:]
    
    def _save_side_metric_plots(self):
        """Create and save plots of running metrics."""
        if self.metrics_df is None:
            return []
        
        # Create a directory for plots
        plots_dir = os.path.join(self.reports_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        plot_files = []
        
        # Plot 1: Foot strike pattern over time
        plt.figure(figsize=(10, 6))
        foot_strike_mapping = {'heel': 1, 'midfoot': 2, 'forefoot': 3}
        numeric_foot_strike = self.metrics_df['strike_pattern'].map(foot_strike_mapping)
        plt.plot(self.metrics_df['frame_number'], numeric_foot_strike, 'o')
        plt.yticks([1, 2, 3], ['Heel', 'Midfoot', 'Forefoot'])
        plt.xlabel('Frame Number')
        plt.ylabel('Foot Strike Pattern')
        plt.title('Foot Strike Pattern Over Time')
        plt.grid(True)
        
        # Save plot
        plot_file = os.path.join(plots_dir, f"{self.session_id}_foot_strike.png")
        plt.savefig(plot_file)
        plot_files.append(plot_file)
        plt.close()
        
        # Plot 2: Trunk angle
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics_df['frame_number'], self.metrics_df['trunk_angle_degrees'], 'b-', label='Trunk Angle')
        # Add reference lines for optimal range
        plt.axhline(y=100 - 90, color='g', linestyle='--', alpha=0.7, label='Optimal')
        plt.axhline(y=100 - 85, color='y', linestyle='--', alpha=0.5, label='Lower Limit')
        plt.axhline(y=100 - 95, color='y', linestyle='--', alpha=0.5, label='Upper Limit')
        plt.xlabel('Frame Number')
        plt.ylabel('Angle (degrees)')
        plt.title('Trunk Angle Over Time')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_file = os.path.join(plots_dir, f"{self.session_id}_trunk_angle.png")
        plt.savefig(plot_file)
        plot_files.append(plot_file)
        plt.close()
        
        # Plot 3: Foot Landing Over Time
        plt.figure(figsize=(10, 6))
        foot_landing_mapping = {'behind': 1, 'under': 2, 'ahead': 3}
        numeric_foot_landing = self.metrics_df['foot_landing_position_category'].map(foot_landing_mapping)
        plt.plot(self.metrics_df['frame_number'], numeric_foot_landing, 'o')
        plt.yticks([1, 2, 3], ['Behind', 'Under', 'Ahead'])
        plt.xlabel('Frame Number')
        plt.ylabel('Foot Landing Pattern')
        plt.title('Foot Landing Pattern Over Time')
        plt.grid(True)
        
        # Save plot
        plot_file = os.path.join(plots_dir, f"{self.session_id}_foot_landing.png")
        plt.savefig(plot_file)
        plot_files.append(plot_file)
        plt.close()
        
        # Plot 4: Knee Angle Comparison (Left vs Right)
        if 'knee_angle_left' in self.metrics_df.columns and 'knee_angle_right' in self.metrics_df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics_df['frame_number'], self.metrics_df['knee_angle_left'], 'b-', label='Left Knee')
            plt.plot(self.metrics_df['frame_number'], self.metrics_df['knee_angle_right'], 'r-', label='Right Knee')
            plt.xlabel('Frame Number')
            plt.ylabel('Knee Angle (degrees)')
            plt.title('Knee Angle Comparison')
            plt.legend()
            plt.grid(True)
            
            # Save plot
            plot_file = os.path.join(plots_dir, f"{self.session_id}_knee_angles.png")
            plt.savefig(plot_file)
            plot_files.append(plot_file)
            plt.close()
        
        # Plot 7: Landing Stiffness Analysis
        if 'strike_landing_stiffness' in self.metrics_df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics_df['frame_number'], self.metrics_df['strike_landing_stiffness'], 'o')
            plt.xlabel('Frame Number')
            plt.ylabel('Stiffness Rating (0-10)')
            plt.title('Landing Stiffness Over Time')
            
            # Add reference bands for optimal zones
            plt.yticks([1, 2, 3], ['not_applicable', 'Stiff', 'Compliant'])
            
            plt.ylim(0, 3)
            plt.legend()
            plt.grid(True)
            
            # Save plot
            plot_file = os.path.join(plots_dir, f"{self.session_id}_landing_stiffness.png")
            plt.savefig(plot_file)
            plot_files.append(plot_file)
            plt.close()

        return plot_files


class RearViewReportGenerator(BaseReportGenerator):
    @property
    def view_name(self):
        return "Rear"

    def _generate_metrics_summary_section(self, html_content):
        # ... (specific logic from current RearReportGenerator._generate_rear_metrics_summary_section)
        if self.metrics_df.empty:
            html_content.append("<div class='section'><h2>Rear Metrics Summary</h2><div class='metric-box'><p>No rear view metrics data available.</p></div></div>")
            return {}

        html_content.extend([
            "        <div class='section'>",
            "            <h2>Rear Metrics Summary</h2>",
            "            <div class='row'>"
        ])

        summary_data = {}

        # Crossover Percentages (Special Calculation)
        for side in ['left', 'right']:
            if f'{side}_foot_crossover' in self.metrics_df.columns and \
            'stance_foot' in self.metrics_df.columns and \
            'stance_phase_detected' in self.metrics_df.columns:
                
                side_stance_frames = self.metrics_df[
                    (self.metrics_df['stance_foot'] == side) &
                    (self.metrics_df['stance_phase_detected'] == True)
                ]
                
                if not side_stance_frames.empty:
                    crossover_frames = side_stance_frames[side_stance_frames[f'{side}_foot_crossover'] == True]
                    crossover_percent = (len(crossover_frames) / len(side_stance_frames)) * 100 if len(side_stance_frames) > 0 else 0
                else:
                    crossover_percent = 0.0 # No stance frames for this side, so 0% crossover

                summary_data[f'{side}_crossover_percent'] = crossover_percent
                
                # Rating for crossover (lower is better, e.g. <5% optimal, <10% good)
                rating_text, rating_class, rating_key = "High", self.RATING_CLASSES["needs-work"], "needs-work"
                if crossover_percent < 5: rating_text, rating_class, rating_key = "Optimal", self.RATING_CLASSES["optimal"], "optimal"
                elif crossover_percent < 10: rating_text, rating_class, rating_key = "Good", self.RATING_CLASSES["good"], "good"
                elif crossover_percent < 20: rating_text, rating_class, rating_key = "Fair", self.RATING_CLASSES["fair"], "fair"

                self._add_metric_box(html_content, f"{side.capitalize()} Foot Crossover", f"{crossover_percent:.1f}%",
                                    rating_text=rating_text, rating_class=rating_class,
                                    progress_percent=(100 - crossover_percent), # Inverted for "good" bar
                                    progress_bar_class_key=rating_key,
                                    sub_text="% of stance phase frames crossing midline.")
            else:
                summary_data[f'{side}_crossover_percent'] = None
                self._add_metric_box(html_content, f"{side.capitalize()} Foot Crossover", "N/A", sub_text="Data not available.")


        # Other Rear Metrics
        def add_summary_metric(col_name, title, unit="", val_format="{:.2f}", std_val_format="{:.2f}", is_categorical=False):
            mean, std, primary, percent = self._get_series_stats(self.metrics_df, col_name)
            
            if is_categorical:
                summary_data[f"{col_name}_primary"] = primary
                summary_data[f"{col_name}_percent"] = percent
                sub_text = f"({percent:.1f}% dominance)" if primary != "N/A" else None
                self._add_metric_box(html_content, title, primary if primary else "N/A", sub_text=sub_text)
            else:
                summary_data[f"{col_name}_mean"] = mean
                summary_data[f"{col_name}_std"] = std
                value_str = val_format.format(mean) if mean is not None else "N/A"
                std_str = std_val_format.format(std) if std is not None else None
                # Add rating logic here if applicable for these metrics
                self._add_metric_box(html_content, title, value_str, unit=unit, std_dev_str=std_str)
        
        # Note: Original 'hip_drop_value' format was .4f for mean and .4 for std. Adjusting to .2f for consistency.
        add_summary_metric('hip_drop_value', "Hip Drop Value", unit="cm", val_format="{:.2f}", std_val_format="{:.2f}")
        add_summary_metric('pelvic_tilt_angle', "Pelvic Tilt Angle", unit="°")
        add_summary_metric('symmetry', "Stride Symmetry", unit="%", val_format="{:.1f}") # Assuming 'symmetry' is numeric %
        add_summary_metric('shoulder_rotation', "Shoulder Rotation", is_categorical=True)

        # Watch Data (can be duplicated from side view if the source df is the same or merged)
        for col, title, unit, fmt in [
            ('vertical_oscillation', "Vertical Oscillation", "cm", "{:.1f}"),
            ('ground_contact_time', "Ground Contact Time", "ms", "{:.0f}"),
            ('stride_length', "Stride Length", "cm", "{:.1f}"), # Assuming this is 'stride_length_cm' from side
            ('cadence', "Cadence", "spm", "{:.0f}")
        ]:
            if col in self.metrics_df.columns: # Check if watch data is in rear_metrics
                add_rear_metric(col, title, unit, fmt)

        html_content.extend(["            </div>", "        </div>"])
        self.summary_data_cache['rear'] = summary_data # Cache for later use (e.g. recommendations)
        return summary_data

    def _generate_plots_section(self, html_content):
        # ... (logic from current RearReportGenerator._generate_rear_plots_section)
        # plot_files = plotting_utils.save_rear_view_plots(self.metrics_df, self.session_id, os.path.join(self.reports_dir, self.plots_sub_dir))
        if self.metrics_df.empty: return

        plot_files = self._save_rear_metric_plots() # Specific to rear view

        if not plot_files:
            html_content.append("<div class='section'><h2>Rear Metrics Visualization</h2><div class='metric-box'><p>No plots generated or available for rear view.</p></div></div>")
            return

        html_content.extend(["        <div class='section'>", "            <h2>Rear Metrics Visualization</h2>"])
        for i, plot_file in enumerate(plot_files):
            if i % 2 == 0:
                if i > 0: html_content.append("            </div>")
                html_content.append("            <div class='row'>")
            
            base_dir = os.path.dirname(self.report_file_path) if self.report_file_path else "."
            rel_path = plot_file
            try:
                if os.path.isabs(plot_file) and os.path.commonprefix([plot_file, os.path.abspath(base_dir)]):
                    rel_path = os.path.relpath(plot_file, base_dir)
            except ValueError: pass

            html_content.extend([
                "                <div class='column'>",
                "                    <div class='chart-container'>",
                f"                        <img src='{rel_path}' alt='Rear View Plot {i+1}' class='chart'>",
                "                    </div>",
                "                </div>"
            ])
        if plot_files: html_content.append("            </div>")
        html_content.append("        </div>")

    def _generate_recommendations_section(self, html_content):
        # ... (logic from current RearReportGenerator._generate_rear_recommendations_section)
        # recommendations = recommendation_engine.generate_rear_view_recommendations(self.metrics_df, self.summary_data_cache.get('rear'))
        if self.metrics_df.empty: return
        
        recommendations = self._generate_rear_recommendations()

        html_content.append("<div class='section'><h2>Rear View Gait Analysis & Recommendations</h2><div class='metric-box' style='min-height:auto; padding: 5px 15px 15px 15px;'>")
        if recommendations:
            html_content.append("<h3>Form Recommendations (Rear View)</h3><ul>")
            for rec in recommendations:
                html_content.append(f"    <li>{rec}</li>")
            html_content.append("</ul>")
        else:
            html_content.append("<p>No specific rear view recommendations generated. Focus on overall balance and symmetry.</p>")
        html_content.append("</div></div>")

    def _generate_rear_recommendations(self):
        """Generate running form recommendations based on metrics."""
        if self.metrics_df is None:
            return []
        
        recommendations = []
        
        # Add general recommendations if list is short
        if len(recommendations) < 3:
            recommendations.append(
                "Maintain a consistent running cadence between 170-180 steps per minute for optimal efficiency. "
                "A metronome app can help you achieve this rhythm."
            )
            
            recommendations.append(
                "Focus on relaxed shoulders and a proper arm swing that moves forward and backward, not across your body. "
                "This helps maintain rotational balance and overall efficiency."
            )
        
        return recommendations


    def _save_rear_metric_plots(self):
        """Create and save plots of running metrics."""
        if self.metrics_df is None:
            return []
        
        # Create a directory for plots
        plots_dir = os.path.join(self.reports_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        plot_files = []
        
        # Plot 1: Foot distance from midline
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics_df['frame_number'], -self.metrics_df['left_distance_from_midline'], 'bo-', label='Left Foot')
        plt.plot(self.metrics_df['frame_number'], -self.metrics_df['right_distance_from_midline'], 'ro-', label='Right Foot')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        plt.xlabel('Frame Number')
        plt.ylabel('Distance from Midline')
        plt.title('Foot Distance from Midline Over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(plots_dir, f"{self.session_id}_foot_distance_midline.png")
        plt.savefig(plot_file)
        plot_files.append(plot_file)
        plt.close()

        # Plot 3: Hip drop with direction indicator
        plt.figure(figsize=(10, 6))
        colors = []
        for direction in self.metrics_df['hip_drop_direction']:
            if direction == 'left':
                colors.append('blue')
            elif direction == 'right':
                colors.append('red')
            else:  # neutral
                colors.append('green')

        plt.bar(self.metrics_df['frame_number'], self.metrics_df['hip_drop_value'], color=colors)
        plt.xlabel('Frame Number')
        plt.ylabel('Hip Drop Value (m)')
        plt.title('Hip Drop with Direction')

        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='Left Drop'),
            Patch(facecolor='red', label='Right Drop'),
            Patch(facecolor='green', label='Neutral')
        ]
        plt.legend(handles=legend_elements)
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(plots_dir, f"{self.session_id}_hip_drop.png")
        plt.savefig(plot_file)
        plot_files.append(plot_file)
        plt.close()

        # Plot 4: Pelvic tilt angle with elevated side indicator
        plt.figure(figsize=(10, 6))
        markers = []
        for side in self.metrics_df['pelvic_tilt_elevated_side']:
            if side == 'left':
                markers.append('^')  # triangle up
            elif side == 'right':
                markers.append('v')  # triangle down
            else:  # neutral
                markers.append('o')  # circle

        for i, (x, y, marker) in enumerate(zip(self.metrics_df['frame_number'], self.metrics_df['pelvic_tilt_angle'], markers)):
            if self.metrics_df['pelvic_tilt_elevated_side'][i] == 'left':
                plt.plot(x, y, marker, color='blue', markersize=10, label='Left Elevated' if i == 0 else "")
            elif self.metrics_df['pelvic_tilt_elevated_side'][i] == 'right':
                plt.plot(x, y, marker, color='red', markersize=10, label='Right Elevated' if i == 1 else "")
            else:  # neutral
                plt.plot(x, y, marker, color='green', markersize=10, label='Neutral' if i == 2 else "")

        plt.plot(self.metrics_df['frame_number'], self.metrics_df['pelvic_tilt_angle'], 'k--', alpha=0.5)
        plt.axhline(y=3, color='g', linestyle='--', alpha=0.7, label='Optimal')
        plt.axhline(y=6, color='y', linestyle='--', alpha=0.5, label='Moderate Limit')
        plt.axhline(y=10, color='r', linestyle='--', alpha=0.5, label='Severe Limit')
        plt.xlabel('Frame Number')
        plt.ylabel('Pelvic Tilt Angle (degrees)')
        plt.title('Pelvic Tilt Angle with Elevated Side')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(plots_dir, f"{self.session_id}_pelvic_tilt.png")
        plt.savefig(plot_file)
        plot_files.append(plot_file)
        plt.close()

        # # Plot 5: Combined biomechanics plot
        # plt.figure(figsize=(12, 8))

        # # Normalize values for comparison
        # max_dist = max(max(abs(min(self.metrics_df['left_distance_from_midline'])), max(self.metrics_df['right_distance_from_midline'])))
        # normalized_left = [x / max_dist for x in self.metrics_df['left_distance_from_midline']]
        # normalized_right = [x / max_dist for x in self.metrics_df['right_distance_from_midline']]
        # normalized_hip = [x / max(abs(min(self.metrics_df['hip_drop_value'])), max(self.metrics_df['hip_drop_value'])) for x in self.metrics_df['hip_drop_value']]
        # normalized_pelvic = [x / max(abs(min(self.metrics_df['pelvic_tilt_angle'])), max(self.metrics_df['pelvic_tilt_angle'])) for x in self.metrics_df['pelvic_tilt_angle']]

        # plt.plot(self.metrics_df['frame_number'], normalized_left, 'b-', label='Left Foot Position (norm)')
        # plt.plot(self.metrics_df['frame_number'], normalized_right, 'r-', label='Right Foot Position (norm)')
        # plt.plot(self.metrics_df['frame_number'], normalized_hip, 'g-', label='Hip Drop (norm)')
        # plt.plot(self.metrics_df['frame_number'], normalized_pelvic, 'y-', label='Pelvic Tilt (norm)')

        # # Add stride phase indicators (assuming frame 2 is mid-stride)
        # plt.axvline(x=2, color='purple', linestyle='--', alpha=0.5, label='Mid-stride')

        # plt.xlabel('Frame Number')
        # plt.ylabel('Normalized Values')
        # plt.title('Combined Running Biomechanics')
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        
        # # Save plot
        # plot_file = os.path.join(plots_dir, f"{self.session_id}_rear_initial_combined.png")
        # plt.savefig(plot_file)
        # plot_files.append(plot_file)
        # plt.close()

        
        return plot_files
    # ... other specific rear view sections ...