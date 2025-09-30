# file: plots.py

import collections
import numpy as np
import matplotlib.pyplot as plt

# Import necessary types from concordia
from concordia.utils import measurements as measurements_lib
from concordia.utils import plotting

def _get_averaged_data_by_timestep(
    measurements: measurements_lib.Measurements,
    channels: list[str],
    agents: list[str],
    metric_name: str = "Metric",
    verbose_print: bool = False,
) -> tuple[list[str], dict[str, list[float]]]:
    """
    Extracts data, groups it by timestep, calculates the average score,
    and optionally prints the underlying data for debugging.
    """
    all_data = []
    available_channels = list(measurements.available_channels())
    for channel_name in channels:
        if channel_name in available_channels:
            try:
                channel_subject = measurements.get_channel(channel_name)
                items = []
                subscription = channel_subject.subscribe(on_next=items.append)
                subscription.dispose()
                all_data.extend(items)
            except Exception as e:
                print(f"  Exception during data retrieval for '{channel_name}': {e}")
        else:
             print(f"Warning: Channel '{channel_name}' not found.")

    if verbose_print and all_data:
        print(f"\n--- Raw Data Points for {metric_name} (before averaging) ---")
        all_data.sort(key=lambda d: (d.get('clock_step', 0), d.get('time_str', '')))
        for datum_idx, datum in enumerate(all_data):
            player = datum.get('player', 'N/A')
            time = datum.get('time_str', 'N/A')
            step = datum.get('clock_step', 'N/A')
            score = datum.get('value_float', np.nan)
            action = datum.get('action_evaluated', '')
            rationale = datum.get('rationale', '')
            print_str = f"  - Raw Datum {datum_idx}: Player={player}, Step={step}, Time={time}, Score={score:.2f}"
            if action:
                print_str += f", Action='{action}'"
            if rationale:
                 print_str += f", Rationale='{rationale}'"
            print(print_str)
    
    if not all_data:
        return [], {}

    grouped_scores = collections.defaultdict(lambda: collections.defaultdict(list))
    timestep_labels = {}
    all_clock_steps = {d.get('clock_step') for d in all_data if d.get('clock_step') is not None}
    
    if not all_clock_steps:
        return [], {}
        
    min_step, max_step = min(all_clock_steps), max(all_clock_steps)
    all_timesteps = range(min_step, max_step + 1)

    for datum in all_data:
        step = datum.get('clock_step')
        player = datum.get('player')
        score = datum.get('value_float')

        if step is not None and player is not None and score is not None:
            grouped_scores[step][player].append(score)
            if step not in timestep_labels:
                timestep_labels[step] = datum.get('time_str', str(step))

    processed_data = {agent: [] for agent in agents}
    final_timestep_labels = []

    for step in all_timesteps:
        final_timestep_labels.append(timestep_labels.get(step, f"Step {step}"))
        for agent in agents:
            agent_scores_for_step = grouped_scores[step].get(agent)
            if agent_scores_for_step:
                average_score = np.mean(agent_scores_for_step)
                processed_data[agent].append(average_score)
            else:
                processed_data[agent].append(np.nan)

    if verbose_print and processed_data:
        print(f"\n--- Averaged Data for {metric_name} Plot ---")
        for i, label in enumerate(final_timestep_labels):
            print(f"  - Timestep {i} (Label: {label}):")
            for agent in agents:
                score = processed_data[agent][i]
                print(f"    - {agent}: Average Score = {score:.2f}")

    return final_timestep_labels, processed_data


def _plot_identity_recall_raw(
    measurements: measurements_lib.Measurements,
    agents: list[str],
    save_to_dir: str = None
):
    """
    Plots the raw, unaveraged Identity Recall scores with verbose data printing.
    This function contains the exact logic from the original main script.
    """
    print("\n--- Generating Raw Identity Recall Plot ---")

    initial_channels_list = list(measurements.available_channels())
    
    ir_channels = [f'{agent}_IdentityRecallMetric' for agent in agents]
    channels_to_plot = [ch for ch in ir_channels if ch in initial_channels_list]

    if channels_to_plot:
        print("\n--- Data for Identity Recall Chart (printing before plotting) ---")

        for channel_name in channels_to_plot:
            print(f"DEBUG: Processing channel: '{channel_name}'")
            channel_data_list = []
            
            try:
                channel_subject = measurements.get_channel(channel_name)
                current_channel_items = []
                subscription = channel_subject.subscribe(
                    on_next=lambda item: current_channel_items.append(item),
                    on_error=lambda err: print(f"  Error processing channel '{channel_name}': {err}")
                )
                subscription.dispose()
                channel_data_list = current_channel_items
            except Exception as e:
                print(f"  Exception during data retrieval for '{channel_name}': {e}")

            print(f"Data for channel: {channel_name}")
            if channel_data_list:
                for datum_idx, datum in enumerate(channel_data_list):
                    player_name = datum.get('player', 'N/A')
                    time_str = datum.get('time_str', 'N/A')
                    recall_score = datum.get('value_float', np.nan)
                    print(f"  Datum {datum_idx}: Player: {player_name}, Time: {time_str}, Score: {recall_score:.4f}")
            else:
                print(f"  No data points found or retrieved for the channel '{channel_name}'.")

        print("\n--- Now proceeding to plot the graph ---")
        try:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            for channel_name_to_plot in channels_to_plot:
                plotting.plot_line_measurement_channel(
                    measurements, 
                    channel_name_to_plot,
                    group_by='player',
                    xaxis='time_str',
                    ax=ax
                )
            ax.set_title('Identity Recall Scores for Alice and Bob (Raw Data)')
            ax.set_xlabel('Time of Action')
            ax.set_ylabel('Identity Recall Score')
            ax.legend(title='Agent')
            ax.grid(True, linestyle='--', alpha=0.6)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            if save_to_dir:
                import os
                os.makedirs(save_to_dir, exist_ok=True)
                plot_path = os.path.join(save_to_dir, "identity_recall_plots.png")
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Identity recall plot saved to {plot_path}")
            plt.show()
        except Exception as e:
            print(f"ERROR during plotting: {e}")
    else:
        print("No channels identified for plotting Identity Recall for Alice or Bob.")


def plot_metrics(measurements: measurements_lib.Measurements, save_to_dir: str = None):
    """
    Generates and displays all metric plots for the simulation.
    If save_to_dir is provided, also saves the plots to that directory.
    """
    # Ensure the output directory exists if saving is requested
    if save_to_dir:
        import os
        os.makedirs(save_to_dir, exist_ok=True)
        print(f"Ensuring output directory exists: {save_to_dir}")
    
    agents_to_plot = ['Alice', 'Bob']


    # --- Plot 1: Identity Recall (Raw Data) ---
    _plot_identity_recall_raw(measurements, agents_to_plot, save_to_dir)
        
    # --- Plot 2: Action Alignment (Averaged per Timestep) ---
    print("\n--- Generating Averaged Action Alignment Plot ---")
    aa_channels = [f'{agent}_action_alignment' for agent in agents_to_plot]
    aa_timestep_labels, aa_processed_data = _get_averaged_data_by_timestep(
        measurements, 
        aa_channels, 
        agents_to_plot,
        metric_name="Action Alignment",
        verbose_print=True
    )

    if aa_processed_data and any(not all(np.isnan(v)) for v in aa_processed_data.values()):
        fig_aa, ax_aa = plt.subplots(figsize=(12, 6))
        for agent_name, scores in aa_processed_data.items():
            ax_aa.plot(aa_timestep_labels, scores, 'o-', label=agent_name)

        ax_aa.set_title('Averaged Action Alignment Scores (per Timestep)')
        ax_aa.set_xlabel('Simulation Timestep')
        ax_aa.set_ylabel('Average Action Alignment Score (1-10)')
        ax_aa.legend(title='Agent')
        ax_aa.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        if save_to_dir:
            import os
            plot_path = os.path.join(save_to_dir, "action_alignment_plots.png")
            fig_aa.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Action alignment plot saved to {plot_path}")
        plt.show()
    else:
        print("No data found to plot Action Alignment.")

