# side_report_generator.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from runnervision_utils.reports.report_generators.base_report_generator import BaseReportGenerator
from runnervision_utils.reports import rating_utils
from runnervision_utils.reports import text_generation # Ensure this is imported

class SideViewReportGenerator(BaseReportGenerator):
    @property
    def view_name(self):
        return "Side"

    # --- 1. METRICS SUMMARY SECTION ---
    # This section now ensures all calculated data is stored in `summary_data` for the LLM.
    def _generate_metrics_summary_section(self, html_content):
        if self.metrics_df.empty:
            html_content.append("<div class='section'><h2>Side Metrics Summary</h2><div class='metric-box'><p>No side view metrics data available.</p></div></div>")
            return {}

        html_content.extend([
            "        <div class='section'>",
            "            <h2>Side View Metrics Summary</h2>",
            "            <div class='row'>"
        ])

        summary_data = {}

        def add_summary_metric(col_name, title, unit="", val_format="{:.1f}", is_categorical=False):
            mean, std, primary, percent = self._get_series_stats(self.metrics_df, col_name)

            if is_categorical:
                summary_data[f"{col_name}_primary"] = primary
                summary_data[f"{col_name}_percent"] = percent
                sub_text = f"({percent:.1f}% dominance)" if primary != "N/A" else None
                self._add_metric_box(html_content, title, primary or "N/A", sub_text=sub_text)
            else:
                summary_data[f"{col_name}_mean"] = mean
                summary_data[f"{col_name}_std"] = std
                value_str = val_format.format(mean) if mean is not None else "N/A"
                std_str = "{:.1f}".format(std) if std is not None else None
                self._add_metric_box(html_content, title, value_str, unit=unit, std_dev_str=std_str)

        # Use the helper to add all original metrics
        add_summary_metric('strike_pattern', "Foot Strike Pattern", is_categorical=True)
        add_summary_metric('cadence', "Cadence", unit=" spm", val_format="{:.0f}")
        add_summary_metric('avg_contact_time_ms', "Ground Contact Time", unit=" ms", val_format="{:.0f}")
        add_summary_metric('vertical_oscillation_cm', "Vertical Oscillation", unit=" cm")
        add_summary_metric('trunk_angle_degrees', "Trunk Forward Lean", unit="°")
        add_summary_metric('knee_angle_left', "Avg. Left Knee Angle", unit="°")
        add_summary_metric('knee_angle_right', "Avg. Right Knee Angle", unit="°")
        add_summary_metric('arm_swing_amplitude', "Arm Swing Amplitude", unit="°")

        html_content.extend(["            </div>", "        </div>"])
        # KEY: Cache the data for use in other sections
        self.summary_data_cache['side'] = summary_data
        return summary_data

    # --- 2. SPECIALIZED SECTIONS ---
    # These sections add more detailed analysis and feed their results back into the summary_data dictionary.
    def _generate_specialized_sections(self, html_content, summary_data):
        self._generate_bilateral_comparison_section(html_content, summary_data)

    def _generate_bilateral_comparison_section(self, html_content, summary_data):
        left_avg = summary_data.get('knee_angle_left_mean')
        right_avg = summary_data.get('knee_angle_right_mean')

        if left_avg is None or right_avg is None:
            return

        diff_abs = abs(left_avg - right_avg)
        denominator = (left_avg + right_avg) / 2
        diff_percent = (diff_abs / denominator) * 100 if denominator != 0 else 0
        
        # KEY CHANGE: Add the calculated difference to summary_data so the LLM can use it.
        summary_data['knee_symmetry_diff_percent'] = diff_percent

        rating_text, rating_key = rating_utils.rate_knee_symmetry(diff_percent)
        rating_class = self.RATING_CLASSES.get(rating_key, "")

        html_content.extend([
            "        <div class='section'>",
            "            <h2>Bilateral Comparison</h2>",
            "            <div class='metric-box' style='padding:20px;'>",
            "                <h3>Knee Angle Symmetry</h3>",
            "                <div class='metric-comparison'>",
            f"                    <div class='metric-comparison-item'><div>Left Knee</div><div class='metric-value metric-value-small'>{left_avg:.1f}°</div></div>",
            "                    <div class='comparison-divider'>vs</div>",
            f"                    <div class='metric-comparison-item'><div>Right Knee</div><div class='metric-value metric-value-small'>{right_avg:.1f}°</div></div>",
            "                </div>",
            f"                <div style='text-align:center; margin-top:15px;'>Difference: {diff_abs:.1f}° ({diff_percent:.1f}%)</div>",
            f"                <div class='rating {rating_class}' style='text-align:center; margin-top:5px;'>{rating_text}</div>",
            "            </div>",
            "        </div>"
        ])

    # --- 3. PLOTS SECTION ---
    # This ensures your original plotting functions are preserved.
    def _generate_plots_section(self, html_content):
        plot_files = self._save_side_metric_plots()

        if not plot_files:
            html_content.append("<div class='section'><h2>Running Metrics Visualization</h2><div class='metric-box'><p>No plots generated or available.</p></div></div>")
            return

        html_content.extend(["<div class='section'><h2>Running Metrics Visualization</h2><div class='row'>"])
        for i, plot_file in enumerate(plot_files):
            rel_path = os.path.join(self.plots_sub_dir, os.path.basename(plot_file))
            html_content.extend([
                "<div class='column'><div class='chart-container'>",
                f"<img src='{rel_path}' alt='Metrics Plot {i+1}' class='chart'>",
                "</div></div>"
            ])
            # Create a new row after every 2 plots
            if (i + 1) % 2 == 0:
                html_content.append("</div><div class='row'>")
        html_content.append("</div></div>") # Close final row and section

    def _save_side_metric_plots(self):
        # This is your original plotting logic, restored to ensure no loss of features.
        # (This code is copied from your original provided script)
        if self.metrics_df is None or self.metrics_df.empty:
            return []
        
        plots_dir = os.path.join(self.reports_dir, self.plots_sub_dir)
        os.makedirs(plots_dir, exist_ok=True)
        plot_files = []
        
        plot_configs = {
            'foot_strike': ('strike_pattern', 'Foot Strike Pattern Over Time', 'Foot Strike Pattern', {'heel': 1, 'midfoot': 2, 'forefoot': 3}),
            'trunk_angle': ('trunk_angle_degrees', 'Trunk Angle Over Time', 'Angle (degrees)', None),
            'knee_angles': (['knee_angle_left', 'knee_angle_right'], 'Knee Angle Comparison', 'Knee Angle (degrees)', None)
        }

        for key, config in plot_configs.items():
            col, title, ylabel, mapping = config
            plt.figure(figsize=(10, 5))
            
            if isinstance(col, list): # Handle comparison plots
                if all(c in self.metrics_df.columns for c in col):
                    plt.plot(self.metrics_df['frame_number'], self.metrics_df[col[0]], 'b-', label='Left Knee')
                    plt.plot(self.metrics_df['frame_number'], self.metrics_df[col[1]], 'r-', label='Right Knee')
                    plt.legend()
            elif col in self.metrics_df.columns:
                if mapping:
                    numeric_series = self.metrics_df[col].map(mapping)
                    plt.plot(self.metrics_df['frame_number'], numeric_series, 'o')
                    plt.yticks(list(mapping.values()), list(mapping.keys()))
                else:
                    plt.plot(self.metrics_df['frame_number'], self.metrics_df[col], 'b-')
            
            plt.title(title)
            plt.xlabel('Frame Number')
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.tight_layout()
            
            plot_file = os.path.join(plots_dir, f"{self.session_id}_{key}.png")
            plt.savefig(plot_file)
            plot_files.append(plot_file)
            plt.close()
            
        return plot_files

    # --- 4. RECOMMENDATIONS SECTION ---
    # This section is now powered by the LLM, replacing the old assessment and recommendation methods.
    def _generate_recommendations_section(self, html_content):
        summary_data = self.summary_data_cache.get('side', {})
        if not summary_data:
            return

        # 1. Get the raw markdown from the new side-view LLM function
        generated_text_markdown = text_generation.generate_side_view_summary_from_llm(summary_data, self.language_model)

        # 2. Convert markdown to HTML
        llm_html_output = generated_text_markdown

        # 3. Append to the report with the overflow-fixing CSS
        html_content.append("<div class='section'>")
        html_content.append(
            "<div class='metric-box' style='min-height:auto; padding: 15px 20px; white-space: pre-wrap; overflow-wrap: break-word;'>"
        )
        html_content.append(llm_html_output)
        html_content.append("</div></div>")

    # The _generate_overall_assessment_section is no longer needed as its role is filled by the LLM summary.
    def _generate_overall_assessment_section(self, html_content, summary_data):
        pass