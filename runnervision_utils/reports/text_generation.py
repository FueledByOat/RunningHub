# text_generation.py

import json
import logging
import markdown
import numpy as np
from utils.language_model_utils import LanguageModel

logger = logging.getLogger(__name__)

def generate_rear_view_summary_from_llm(summary_data, lm_model):
    """
    Generates a text summary for the rear view analysis using a local language model
    and formats the markdown response into HTML.
    """
    # 1. Select and format metrics for the model
    llm_input_data = {
        "Left Foot Crossover Percent of Session": summary_data.get('left_crossover_percent'),
        "Right Foot Crossover Percent of Session": summary_data.get('right_crossover_percent'),
        "Avg Hip Drop (cm)": summary_data.get('hip_drop_value_mean'),
        "Avg Pelvic Tilt (degrees)": summary_data.get('pelvic_tilt_angle_mean'),
        # "Stride Symmetry (%)": summary_data.get('symmetry_mean'),
        "Dominant Shoulder Rotation": summary_data.get('shoulder_rotation_primary'),
        # "Cadence (spm)": summary_data.get('cadence_mean'),
        # "Vertical Oscillation (cm)": summary_data.get('vertical_oscillation_mean'),
    }
    # Filter out unavailable metrics and format numbers nicely
    llm_input_data = {
        k: f"{v:.1f}" if isinstance(v, (float, np.floating)) else v
        for k, v in llm_input_data.items() if v is not None and v != "N/A"
    }

    if not llm_input_data:
        return "<p>Insufficient data to generate an AI-powered summary.</p>"

    # 2. Construct a prompt that encourages markdown use
    prompt = f"""
    You are an expert running biomechanics coach.
    Based on the following data from a recorded running session, provide an analysis with a title, a short summary,
    and a bulleted list of recommendations, using clinical physical therapist language.
    Use markdown for formatting (### for the heading, - for list items, ** for bold).
    Do not include any details after the conclusion of the bulleted list.

    Runner's Data:
    {json.dumps(llm_input_data, indent=2)}
    """

    # 3. **[Placeholder]** Call your local language model
    try:
        # generated_text = language_model._generate_response(prompt, "", False)
        generated_text = lm_model._generate_report_summary_response(prompt)
    except Exception as e:
        logger.warning(f"Language Model Failed to Generate Report Text. Using Fake Data: {e}")
        generated_text = (
        "Your rear-view analysis indicates some lateral movement, specifically with hip drop and foot crossover. "
        "Focusing on core and hip stability will be key to improving your form and efficiency.\n"
        "- Work on strengthening your gluteus medius with exercises like clamshells and side leg raises to reduce hip drop.\n"
        "- Be mindful of your foot placement, aiming for your feet to land more directly under your hips rather than crossing the midline.\n"
        "- Engage your core throughout your run to maintain a stable pelvis and reduce unnecessary upper body rotation."
    )

    # The rest of the function works perfectly with this output.
    # html_output = markdown.markdown(generated_text)

    return markdown.markdown(generated_text)

def generate_side_view_summary_from_llm(summary_data, lm_model):
    """
    Generates a text summary for the side view analysis using a local language model.

    Args:
        summary_data (dict): A dictionary of summarized metrics for the side view.

    Returns:
        str: A raw markdown string with the generated analysis and recommendations.
    """
    # 1. Select and format the most important side-view metrics for the model
    llm_input_data = {
        "Primary Foot Strike": summary_data.get('strike_pattern_primary'),
        "Cadence (spm)": summary_data.get('cadence_mean'),
        "Ground Contact Time (ms)": summary_data.get('avg_contact_time_ms_mean'),
        "Vertical Oscillation (cm)": summary_data.get('vertical_oscillation_cm_mean'),
        "Trunk Forward Lean (degrees)": summary_data.get('trunk_angle_degrees_mean'),
        "Knee Angle Symmetry Difference (%)": summary_data.get('knee_symmetry_diff_percent'),
        # "Arm Swing Amplitude (degrees)": summary_data.get('arm_swing_amplitude_mean')
    }
    # Filter out unavailable metrics and format numbers nicely
    llm_input_data = {
        k: f"{v:.1f}" if isinstance(v, (float, np.floating)) else v
        for k, v in llm_input_data.items() if v is not None and v != "N/A"
    }

    if not llm_input_data:
        return "### Side-View Analysis\n\n<p>Insufficient data to generate an AI-powered summary.</p>"

    # 2. Construct a clear, side-view-specific prompt
    prompt = f"""
    You are an expert running biomechanics coach analyzing a runner's side-view gait.
    Based on the following data, provide a concise analysis with a title, a short summary paragraph,
    and a bulleted list of the top 2-3 most critical recommendations, using clinical physical therapist language.
    Use markdown for formatting (# for a heading, - for list items, ** for bold).
    Do not include any details after the conclusion of the bulleted list.

    **Runner's Side-View Data:**
    {json.dumps(llm_input_data, indent=2)}
    """

    # 3. **[Placeholder]** Call your local language model's dedicated report function
    generated_text_markdown = lm_model._generate_report_summary_response(prompt)
    #
    # For this example, we'll use a hardcoded response that uses markdown.
    # generated_text_markdown = (
    #     "### Side-View Gait Analysis\n\n"
    #     "Overall, your form shows a solid foundation with a good cadence, which is key for efficiency. "
    #     "The primary areas for refinement are your foot strike, which is heel-dominant, and a slight asymmetry in knee angle. "
    #     "Focusing on these aspects can help reduce braking forces and improve balance.\n\n"
    #     "**Key Recommendations:**\n"
    #     "* **Transition to Midfoot Strike:** Your current heel strike can increase impact forces. Try drills like "
    #     "running in place or short, quick strides, focusing on landing with your foot more underneath your body.\n"
    #     "* **Improve Bilateral Symmetry:** The difference in your knee angles suggests a potential strength imbalance. "
    #     "Incorporate unilateral exercises like single-leg squats and lunges to build equal strength on both sides.\n"
    #     "* **Maintain Trunk Posture:** Your forward lean is in a good range. Continue to focus on keeping your core "
    #     "engaged to maintain this strong, stable posture throughout your runs."
    # )

    # 4. Return the raw markdown for the generator to process
    return markdown.markdown(generated_text_markdown)