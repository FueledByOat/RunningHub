# rear_report_generator.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from runnervision_utils.reports.report_generators.base_report_generator import BaseReportGenerator
from runnervision_utils.reports import rating_utils
from runnervision_utils.reports import text_generation # Import the new text generation utility

class RearViewReportGenerator(BaseReportGenerator):
    @property
    def view_name(self):
        return "Rear"

    def _generate_metrics_summary_section(self, html_content):
        if self.metrics_df.empty:
            html_content.append("<div class='section'><h2>Rear Metrics Summary</h2><div class='metric-box'><p>No rear view metrics data available.</p></div></div>")
            return {}

        html_content.extend([
            "        <div class='section'>",
            "            <h2>Rear View Metrics Summary</h2>",
            "            <div class='row'>"
        ])

        summary_data = {}

        # Crossover Percentages (from previous step)
        for side in ['left', 'right']:
            col_name = f'{side}_foot_crossover'
            if col_name in self.metrics_df.columns:
                crossover_series = self.metrics_df[self.metrics_df['stance_foot'] == side][col_name].dropna()
                percent = crossover_series.mean() * 100 if not crossover_series.empty else 0
                summary_data[f'{side}_crossover_percent'] = percent
                rating_text, rating_key = rating_utils.rate_crossover(percent)
                progress = 100 - min(percent, 100)
                self._add_metric_box(html_content, f"{side.capitalize()} Crossover", f"{percent:.1f}%",
                                     rating_text=rating_text, rating_key=rating_key,
                                     progress_percent=progress,
                                     sub_text="% of stance foot crossing midline.")
            else:
                self._add_metric_box(html_content, f"{side.capitalize()} Crossover", "N/A")


        # Helper to add other metrics back in
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
                std_str = val_format.format(std) if std is not None else None
                self._add_metric_box(html_content, title, value_str, unit=unit, std_dev_str=std_str)

        # === ADDING METRICS BACK IN ===
        add_summary_metric('hip_drop_value', "Avg. Hip Drop", unit="cm")
        add_summary_metric('pelvic_tilt_angle', "Avg. Pelvic Tilt", unit="°")
        add_summary_metric('symmetry', "Stride Symmetry", unit="%", val_format="{:.1f}")
        add_summary_metric('shoulder_rotation', "Shoulder Rotation", is_categorical=True)

        # Add watch data if available
        for col, title, unit, fmt in [
            ('vertical_oscillation', "Vertical Oscillation", "cm", "{:.1f}"),
            ('ground_contact_time', "Ground Contact Time", "ms", "{:.0f}"),
            ('stride_length', "Stride Length", "cm", "{:.1f}"),
            ('cadence', "Cadence", "spm", "{:.0f}")
        ]:
            if col in self.metrics_df.columns:
                add_summary_metric(col, title, unit, fmt)


        html_content.extend(["            </div>", "        </div>"])
        # Ensure the summary data is cached for the recommendation section
        self.summary_data_cache['rear'] = summary_data
        return summary_data


    def _generate_specialized_sections(self, html_content, summary_data):
        """No specialized sections for rear view in this version."""
        pass

    def _generate_recommendations_section(self, html_content):
        """Generates recommendations using the Language Model."""
        if self.metrics_df.empty:
            return

        # Use the data cached by _generate_metrics_summary_section
        summary_data = self.summary_data_cache.get('rear', {})
        if not summary_data:
            return
        llm_html_output = text_generation.generate_rear_view_summary_from_llm(summary_data, self.language_model)
        
#         llm_html_output = markdown.markdown("""# Example Markdown
# - Bullet point 1
# - Bullet point 2""")
        html_content.append(
            "<div class='section'>"
            "<h2>Rear View Gait Analysis</h2>"
            """<div class='metric-box' style='min-height:auto; padding: 5px 15px 15px 15px;'
            max-width: 100%; 
            word-wrap: break-word; 
            overflow-wrap: break-word;
            overflow: auto; >"""
            f"{llm_html_output}"
            "</div></div>"
        )

    def _generate_overall_assessment_section(self, html_content, summary_data):
        """Overall assessment can be integrated or removed as needed."""
        pass


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