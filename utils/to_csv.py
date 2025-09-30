import csv
import os
from concordia.utils import measurements as measurements_lib

def export_metrics_to_csvs(
    measurements: measurements_lib.Measurements,
    player_names: list[str],
    output_dir: str = "experiment_output"
):
    """
    Gathers raw data from all custom metrics and exports it to two separate, 
    clean CSV files (one per metric type).

    Args:
        measurements: The Measurements object containing all simulation data.
        player_names: A list of player names to gather metrics for.
        output_dir: The directory where the CSV files will be saved.
    """
    print(f"\n--- Exporting raw metrics data to directory: {output_dir} ---")
    os.makedirs(output_dir, exist_ok=True)

    available_channels = list(measurements.available_channels())

    # Helper function to write a list of dictionaries to a CSV file.
    def _write_csv(filename, fieldnames, rows):
        if not rows:
            print(f"No data found for {filename}. Skipping file creation.")
            return
        
        # Sort data by timestamp and player for readability
        rows.sort(key=lambda r: (r.get('timestamp', ''), r.get('player', '')))
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"Successfully wrote {len(rows)} rows to {filename}")
        except IOError as e:
            print(f"Error writing to file {filename}: {e}")

    # 1. Process and Write Identity Recall Data
    # -------------------------------------------
    identity_recall_rows = []
    id_recall_channels = [f'{name}_IdentityRecallMetric' for name in player_names]
    id_recall_fields = ['timestamp', 'clock_step', 'player', 'overall_quiz_score', 'question', 'agent_answer', 'correct_answer', 'question_score']

    for channel_name in id_recall_channels:
        if channel_name in available_channels:
            channel_data = []
            subscription = measurements.get_channel(channel_name).subscribe(on_next=channel_data.append)
            subscription.dispose()

            for datum in channel_data:
                for question_detail in datum.get('individual_scores_details', []):
                    identity_recall_rows.append({
                        'timestamp': datum.get('time_str', ''),
                        'clock_step': datum.get('timestep', ''),
                        'player': datum.get('player', ''),
                        'overall_quiz_score': datum.get('value_float', ''),
                        'question': question_detail.get('question_text', ''),
                        'agent_answer': question_detail.get('agent_answer_provided', ''),
                        'correct_answer': question_detail.get('correct_answer_from_chronicle', ''),
                        'question_score': question_detail.get('score_for_question', '')
                    })
    
    _write_csv(os.path.join(output_dir, 'identity_recall.csv'), id_recall_fields, identity_recall_rows)


    # 2. Process and Write Action Alignment Data
    # ------------------------------------------
    action_alignment_rows = []
    action_align_channels = [f'{name}_action_alignment' for name in player_names]
    action_align_fields = ['timestamp', 'clock_step', 'player', 'alignment_score', 'rationale', 'action_evaluated']

    for channel_name in action_align_channels:
        if channel_name in available_channels:
            channel_data = []
            subscription = measurements.get_channel(channel_name).subscribe(on_next=channel_data.append)
            subscription.dispose()

            for datum in channel_data:
                action_alignment_rows.append({
                    'timestamp': datum.get('time_str', ''),
                    'clock_step': datum.get('clock_step', ''),
                    'player': datum.get('player', ''),
                    'alignment_score': datum.get('value_float', ''),
                    'rationale': datum.get('rationale', ''),
                    'action_evaluated': datum.get('action_evaluated', '')
                })

    _write_csv(os.path.join(output_dir, 'action_alignment.csv'), action_align_fields, action_alignment_rows)


